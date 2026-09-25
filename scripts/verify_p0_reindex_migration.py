"""Pre-deploy verification for the P0-1/P0-2 knowledge-base fixes.

Read-only except for the one safe, additive, idempotent step every deployment
of this codebase already runs before any release: `create_indexes.py`'s
`create_standard_indexes()` (adds `document_status_filter` on
`embeddings_metadata` and the `uniq_reindex_lock` unique-sparse index on
`document_versions` -- both no-ops if they already exist, and neither touches
existing documents).

Why no data-migration step is needed, verified live against THIS
deployment's own data rather than assumed:

  * P0-2 makes every retrieval path (`MongoVectorStore.search`/
    `find_by_section_number`/`find_constitution_article`/`find_named_section`,
    `BM25Index.search`) require `metadata.document_status == "active"`
    unconditionally. `IndexingPipeline.index_file` has stamped this field on
    every chunk it ever wrote since long before this fix -- always
    `"active"` (the bug was that it was NEVER anything else, not that it was
    ever absent). If EVERY existing chunk already carries `"active"`,
    nothing needs backfilling; if some don't (an older schema version, a
    hand-inserted document, a different ingestion path), they would
    silently stop being retrievable the moment this deploys -- worth
    knowing about *before* deploying, not after.
  * `DocumentVersionRepository.latest_for_source` now excludes
    `document_status="staging"` rows. That status did not exist before this
    fix, so no existing version row can carry it -- confirmed by counting
    what values actually occur.
  * The new `uniq_reindex_lock` index is sparse: it only constrains
    documents that HAVE a `reindex_lock_key` field, which, again, no
    existing row does.

Usage:
    python -m scripts.verify_p0_reindex_migration            # report only
    python -m scripts.verify_p0_reindex_migration --apply-indexes
        # also creates the two new indexes (safe to run repeatedly; this is
        # the one step required before deploying)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.models.collections import DOCUMENT_VERSIONS, EMBEDDINGS_METADATA


async def _status_counts(collection: Any, field: str) -> dict[str, int]:
    pipeline = [{"$group": {"_id": f"${field}", "count": {"$sum": 1}}}]
    counts: dict[str, int] = {}
    async for row in collection.aggregate(pipeline):
        key = row["_id"] if row["_id"] is not None else "<missing>"
        counts[str(key)] = row["count"]
    return counts


async def report() -> dict[str, Any]:
    chunks = mongodb.db[EMBEDDINGS_METADATA]
    versions = mongodb.db[DOCUMENT_VERSIONS]

    chunk_status = await _status_counts(chunks, "metadata.document_status")
    version_status = await _status_counts(versions, "document_status")
    reindex_locked = await versions.count_documents({"reindex_lock_key": {"$exists": True}})

    total_chunks = sum(chunk_status.values())
    non_active_chunks = total_chunks - chunk_status.get("active", 0)

    return {
        "total_chunks": total_chunks,
        "chunk_document_status_counts": chunk_status,
        "chunks_that_will_become_unretrievable": non_active_chunks,
        "version_document_status_counts": version_status,
        "pre_existing_staging_versions": version_status.get("staging", 0),
        "rows_currently_holding_the_reindex_lock": reindex_locked,
        "safe_to_deploy_with_no_backfill": non_active_chunks == 0 and version_status.get("staging", 0) == 0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply-indexes", action="store_true",
        help="Also create the two new indexes (document_status_filter, uniq_reindex_lock). Safe to re-run.",
    )
    args = parser.parse_args()

    await mongodb.connect()
    try:
        findings = await report()
        print("P0-1/P0-2 pre-deploy report")
        print(f"  total chunks: {findings['total_chunks']}")
        print(f"  chunk document_status counts: {findings['chunk_document_status_counts']}")
        print(f"  version document_status counts: {findings['version_document_status_counts']}")
        print(f"  rows currently holding the reindex lock: {findings['rows_currently_holding_the_reindex_lock']}")

        if findings["chunks_that_will_become_unretrievable"]:
            print(
                f"\n  WARNING: {findings['chunks_that_will_become_unretrievable']} chunk(s) do NOT have "
                "metadata.document_status=\"active\" and will become UNRETRIEVABLE the moment this deploys. "
                "Investigate before deploying -- this script intentionally does not guess a fix."
            )
        if findings["pre_existing_staging_versions"]:
            print(
                f"\n  WARNING: {findings['pre_existing_staging_versions']} document_versions row(s) already have "
                "document_status=\"staging\" -- these predate this fix and were never a real in-flight reindex; "
                "review with DocumentVersionRepository.find_stale_staging() before deploying."
            )
        if findings["safe_to_deploy_with_no_backfill"]:
            print("\n  No backfill required: every existing chunk/version already satisfies the new invariants.")

        if args.apply_indexes:
            from scripts.create_indexes import create_standard_indexes

            print("\nCreating/verifying indexes...")
            await create_standard_indexes()
            print("Done.")
        else:
            print("\nPass --apply-indexes to create the two new required indexes (safe to re-run).")
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
