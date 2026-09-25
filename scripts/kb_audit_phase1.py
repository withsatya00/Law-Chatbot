"""KB Cleanup Phase 1: Inventory and Safe Backfill.

Read-only reconciliation of `storage/knowledge_base` (disk) against MongoDB
(`uploaded_documents`, `document_versions`, `embeddings_metadata`) and the KB
staging ledger (`kb_staging_records`), followed by the existing jurisdiction
backfill (`scripts/backfill_kb_jurisdiction.py`) run dry-run-first and applied
only if the dry run is internally consistent and the target is a local/dev
database. Nothing here indexes, re-embeds, deletes, moves or renames any KB
file; the backfill is metadata-only (see that script's own docstring).

Reuses rather than reimplements:
  - `DocumentQualityChecker.hash_file` / `KnowledgeBaseIngestionService.
    _corruption_reason` for hashing and the readability probe.
  - `scripts.kb_exact_deduplicate.build_manifest` for exact-duplicate grouping.
  - `app.rag.incremental.IncrementalIndexPlanner.plan` (read-only) for
    new/updated/unchanged/deleted-vs-ledger classification.
  - `scripts.backfill_kb_jurisdiction.backfill` for the actual metadata write.

Usage:
    python scripts/kb_audit_phase1.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache.redis_client import redis_client
from app.cache.response_cache import CACHE_KEY_PREFIX
from app.core.config import settings
from app.core.constants import ALLOWED_UPLOAD_EXTENSIONS
from app.database.mongodb import mongodb
from app.models.collections import (
    DOCUMENT_VERSIONS,
    EMBEDDINGS_METADATA,
    KB_STAGING_RECORDS,
    UPLOADED_DOCUMENTS,
)
from app.rag.incremental import IncrementalIndexPlanner
from app.rag.kb_jurisdiction import shared_retrieval_filters
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from scripts.backfill_kb_jurisdiction import backfill as run_jurisdiction_backfill
from scripts.kb_exact_deduplicate import build_manifest as build_dedupe_manifest

AUDIT_DIR = Path(__file__).resolve().parent.parent / "storage" / "kb_audit"


async def build_inventory() -> dict[str, Any]:
    root = settings.knowledge_base_dir.resolve()
    svc = KnowledgeBaseIngestionService()
    files = sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.name.casefold())

    dedupe = await build_dedupe_manifest()
    group_sha_of: dict[str, str] = {}
    for plan in dedupe["plans"]:
        for name in [plan["keep"], *plan["archive"]]:
            group_sha_of[name] = plan["sha256"]
    archive_names = {name for plan in dedupe["plans"] for name in plan["archive"]}
    for conflict in dedupe["conflict_groups"]:
        for name in conflict["files"]:
            group_sha_of[name] = conflict["sha256"]

    planner = IncrementalIndexPlanner()
    change_set = await planner.plan(root, ALLOWED_UPLOAD_EXTENSIONS)
    new_names = {p.name for p in change_set.new_files}
    updated_names = {p.name for p in change_set.updated_files}

    documents_coll = mongodb.db[UPLOADED_DOCUMENTS]
    versions_coll = mongodb.db[DOCUMENT_VERSIONS]
    chunks_coll = mongodb.db[EMBEDDINGS_METADATA]
    staging_coll = mongodb.db[KB_STAGING_RECORDS]

    # Same hash set `IndexingPipeline.index_file` itself checks new content
    # against (`_known_hashes`) -- content already indexed under a DIFFERENT,
    # possibly no-longer-present filename is still an exact duplicate, not a
    # genuinely new document. `document_hash` -> one active `source_document`.
    known_hash_owner: dict[str, str] = {}
    async for version in versions_coll.find(
        {"document_status": {"$nin": ["deleted", "superseded"]}},
        {"document_hash": 1, "source_document": 1},
    ):
        doc_hash = version.get("document_hash")
        if doc_hash and doc_hash not in known_hash_owner:
            known_hash_owner[doc_hash] = version.get("source_document")

    # Same content under ANY status (including superseded/deleted), regardless
    # of filename -- evidence only, never used to classify: a hash match
    # against a version that was later deleted is "this content was ingested
    # before, then removed" (a prior-ingestion/deletion question for an admin
    # to resolve), not proof the current file is a live duplicate.
    historical_hash_matches: dict[str, list[dict[str, Any]]] = {}
    async for version in versions_coll.find(
        {}, {"document_hash": 1, "source_document": 1, "document_status": 1, "version_number": 1}
    ):
        doc_hash = version.get("document_hash")
        if not doc_hash:
            continue
        historical_hash_matches.setdefault(doc_hash, []).append(
            {
                "source_document": version.get("source_document"),
                "document_status": version.get("document_status"),
                "version_number": version.get("version_number"),
            }
        )

    physical_names = {p.name for p in files}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for path in files:
        name = path.name
        sha256 = svc.quality.hash_file(path)
        corrupt_reason = svc._corruption_reason(path)

        doc_record = await documents_coll.find_one({"filename": name})
        version_record = await versions_coll.find_one({"source_document": name}, sort=[("version_number", -1)])
        chunk_count = await chunks_coll.count_documents({"metadata.source_document": name})
        staging_record = await staging_coll.find_one(
            {"$or": [{"original_filename": name}, {"content_hash": sha256}]}
        )

        owner_present = False
        for src in (doc_record, version_record):
            if not src:
                continue
            meta = src.get("metadata") or {}
            if src.get("owner_session_id") or src.get("owner_user_id"):
                owner_present = True
            if meta.get("owner_session_id") or meta.get("owner_user_id"):
                owner_present = True
        ownership_scope = "private_owned" if owner_present else "shared_kb"

        metadata = (doc_record or {}).get("metadata") or {}
        review_status = metadata.get("review_status", "legacy_missing_field")
        verification_status = metadata.get("verification_status", "legacy_missing_field")
        hash_matches_version = bool(version_record and version_record.get("document_hash") == sha256)
        content_drifted = bool(
            version_record and version_record.get("document_hash") not in (None, sha256)
        )
        # Content already indexed and active under a DIFFERENT source_document
        # name (commonly one no longer physically present, e.g. an admin
        # ingestion's content-derived/UUID filename) -- a true duplicate even
        # though nothing in the folder itself shares this hash.
        content_duplicate_of = known_hash_owner.get(sha256)
        if content_duplicate_of == name:
            content_duplicate_of = None
        historical_matches = [
            m for m in historical_hash_matches.get(sha256, []) if m["source_document"] != name
        ]

        if corrupt_reason:
            classification = "corrupt_or_unreadable"
        elif name in archive_names:
            classification = "exact_duplicate"
        elif (
            version_record
            and version_record.get("document_status") in ("active", "superseded")
            and chunk_count > 0
            and hash_matches_version
        ):
            classification = "indexed"
        elif version_record and (chunk_count == 0 or content_drifted) or not version_record and chunk_count > 0:
            classification = "partially_indexed"
        elif not version_record and chunk_count == 0 and content_duplicate_of:
            classification = "exact_duplicate"
        elif not version_record and chunk_count == 0 and name in new_names:
            classification = "unindexed"
        else:
            classification = "unrelated_or_needs_manual_review"

        if classification == "unindexed" and historical_matches:
            reconciliation_note = (
                "Content matches a prior document_versions record now "
                f"{historical_matches[0]['document_status']} (source_document="
                f"{historical_matches[0]['source_document']!r}) -- was previously indexed "
                "and later removed/superseded under a different filename. Needs an admin "
                "decision on WHY it was removed before this copy is staged, not an automatic re-index."
            )
        elif classification == "unindexed":
            reconciliation_note = "No indexing history under any filename or content hash: a genuinely new, never-indexed file."
        else:
            reconciliation_note = None

        row = {
            "filename": name,
            "relative_path": name,
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
            "file_type": path.suffix.lower(),
            "readable": corrupt_reason is None,
            "corrupt_reason": corrupt_reason,
            "document_id": str(doc_record["_id"]) if doc_record else None,
            "version_id": str(version_record["_id"]) if version_record else None,
            "chunk_count": chunk_count,
            "review_status": review_status,
            "verification_status": verification_status,
            "index_status": (version_record or {}).get("index_status"),
            "document_status": (version_record or {}).get("document_status"),
            "ownership_scope": ownership_scope,
            "duplicate_group_sha256": group_sha_of.get(name),
            "content_duplicate_of_indexed_source": content_duplicate_of,
            "historical_hash_matches": historical_matches,
            "reconciliation_note": reconciliation_note,
            "staging_ledger_status": (staging_record or {}).get("status"),
            "staging_ledger_id": str(staging_record["_id"]) if staging_record else None,
            "planner_state": "new" if name in new_names else ("updated" if name in updated_names else "unchanged"),
            "classification": classification,
        }
        rows.append(row)
        counts[classification] = counts.get(classification, 0) + 1

    # Orphan document/chunk records: DB rows referencing a filename that is no
    # longer physically present under knowledge_base_dir. Reported, never
    # deleted.
    orphan_doc_names = sorted(
        {
            d["filename"]
            async for d in documents_coll.find(
                {
                    "filename": {"$nin": sorted(physical_names)},
                    "owner_session_id": {"$in": [None]},
                    "owner_user_id": {"$in": [None]},
                },
                {"filename": 1},
            )
        }
    )
    orphan_chunk_names = sorted(
        name
        for name in await chunks_coll.distinct(
            "metadata.source_document",
            {
                "metadata.source_document": {"$nin": sorted(physical_names)},
                "metadata.owner_user_id": {"$in": [None]},
            },
        )
        if name
    )

    total_chunks = await chunks_coll.count_documents({})
    staging_status_counts = {
        status: await staging_coll.count_documents({"status": status})
        for status in ("pending", "processing", "indexed", "duplicate", "failed", "needs_review")
    }

    return {
        "generated_at_root": str(root),
        "total_files": len(files),
        "classification_counts": counts,
        "total_chunks_in_db": total_chunks,
        "orphan_document_records": orphan_doc_names,
        "orphan_chunk_source_documents": orphan_chunk_names,
        "staging_ledger_status_counts": staging_status_counts,
        "duplicate_groups": len(dedupe["plans"]),
        "duplicate_conflict_groups": len(dedupe["conflict_groups"]),
        "files": rows,
    }


async def cache_generation() -> int | None:
    try:
        raw = await redis_client.client.get(f"{CACHE_KEY_PREFIX}:kb-generation")
        return int(raw) if raw is not None else 0
    except Exception:  # noqa: BLE001 - offline audit reports unavailable Redis as unknown
        return None


def _safety_check() -> dict[str, Any]:
    reasons: list[str] = []
    uri = settings.mongodb_uri
    is_local = any(host in uri for host in ("localhost", "127.0.0.1"))
    if not is_local:
        reasons.append(f"mongodb_uri does not look local/dev: {uri!r}")
    if settings.environment not in ("development", "test"):
        reasons.append(f"environment={settings.environment!r} is not development/test")
    return {
        "mongodb_uri_is_local": is_local,
        "environment": settings.environment,
        "safe_to_apply": not reasons,
        "blockers": reasons,
    }


async def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    await mongodb.connect()
    await redis_client.connect()

    print("Building canonical inventory (read-only)...")
    inventory = await build_inventory()
    (AUDIT_DIR / "phase1_inventory.json").write_text(json.dumps(inventory, indent=2, default=str), encoding="utf-8")
    print(f"  total_files={inventory['total_files']} chunks_in_db={inventory['total_chunks_in_db']}")
    print(f"  classification_counts={inventory['classification_counts']}")

    safety = _safety_check()
    print(f"Safety check: {safety}")

    print("Running jurisdiction backfill dry run (read-only, existing script)...")
    dry_run = await run_jurisdiction_backfill(apply=False)
    backfill_dry_run_report = {"safety_check": safety, "dry_run": dry_run}
    (AUDIT_DIR / "backfill_dry_run.json").write_text(
        json.dumps(backfill_dry_run_report, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"  documents_scanned={dry_run['documents_scanned']} "
        f"needs_review={dry_run['documents_marked_needs_review']} "
        f"approved={dry_run['documents_marked_approved']} "
        f"errors={dry_run['documents_with_errors']}"
    )

    verification: dict[str, Any] = {"safety_check": safety}

    if not safety["safe_to_apply"]:
        verification["applied"] = False
        verification["reason"] = "Safety check failed; see blockers."
        (AUDIT_DIR / "post_apply_verification.json").write_text(
            json.dumps(verification, indent=2, default=str), encoding="utf-8"
        )
        print("NOT APPLYING: safety check failed. See post_apply_verification.json for blockers.")
        await mongodb.close()
        return

    if dry_run["documents_with_errors"]:
        verification["applied"] = False
        verification["reason"] = "Dry run reported pre-existing-metadata validation errors; needs admin review first."
        (AUDIT_DIR / "post_apply_verification.json").write_text(
            json.dumps(verification, indent=2, default=str), encoding="utf-8"
        )
        print("NOT APPLYING: dry run has validation errors. See post_apply_verification.json.")
        await mongodb.close()
        return

    gen_before = await cache_generation()
    print(f"Applying jurisdiction backfill (--apply equivalent). cache_generation_before={gen_before}")
    apply_result = await run_jurisdiction_backfill(apply=True)
    gen_after = await cache_generation()

    print("Re-running dry run to confirm idempotency (expect documents_scanned=0)...")
    idempotent_check = await run_jurisdiction_backfill(apply=False)

    print("Rebuilding inventory after apply (files/db counts must be unchanged)...")
    post_inventory = await build_inventory()

    # Private-document isolation: no document with an owner field may have
    # been touched by the backfill (it excludes them at the query stage; this
    # re-derives the same guarantee independently, from the DB, not from the
    # script's own claim).
    documents_coll = mongodb.db[UPLOADED_DOCUMENTS]
    owned_touched = await documents_coll.count_documents(
        {
            "$or": [{"owner_session_id": {"$ne": None}}, {"owner_user_id": {"$ne": None}}],
            "metadata.jurisdiction_schema_version": {"$exists": True},
        }
    )

    chunks_coll = mongodb.db[EMBEDDINGS_METADATA]
    approved_filter = shared_retrieval_filters()
    needs_review_in_shared_filter = await chunks_coll.count_documents(
        {**approved_filter, "metadata.review_status": "needs_review"}
    )
    total_needs_review_chunks = await chunks_coll.count_documents({"metadata.review_status": "needs_review"})
    total_approved_chunks = await chunks_coll.count_documents({"metadata.review_status": "approved"})

    verification.update(
        {
            "applied": True,
            "cache_generation_before": gen_before,
            "cache_generation_after": gen_after,
            "cache_generation_bumped": (gen_after is not None and gen_before is not None and gen_after > gen_before),
            "apply_result": apply_result,
            "idempotent_second_run": {
                "documents_scanned": idempotent_check["documents_scanned"],
                "is_idempotent": idempotent_check["documents_scanned"] == 0,
            },
            "counts_before_apply": {
                "total_files": inventory["total_files"],
                "total_chunks_in_db": inventory["total_chunks_in_db"],
                "classification_counts": inventory["classification_counts"],
            },
            "counts_after_apply": {
                "total_files": post_inventory["total_files"],
                "total_chunks_in_db": post_inventory["total_chunks_in_db"],
                "classification_counts": post_inventory["classification_counts"],
            },
            "folder_db_chunk_counts_unchanged": (
                inventory["total_files"] == post_inventory["total_files"]
                and inventory["total_chunks_in_db"] == post_inventory["total_chunks_in_db"]
            ),
            "private_documents_touched_by_backfill": owned_touched,
            "private_isolation_ok": owned_touched == 0,
            "shared_retrieval_excludes_needs_review": {
                "needs_review_chunks_passing_shared_filter": needs_review_in_shared_filter,
                "total_needs_review_chunks": total_needs_review_chunks,
                "total_approved_chunks": total_approved_chunks,
                "ok": needs_review_in_shared_filter == 0,
            },
        }
    )
    (AUDIT_DIR / "post_apply_verification.json").write_text(
        json.dumps(verification, indent=2, default=str), encoding="utf-8"
    )
    print("Wrote post_apply_verification.json")
    print(f"  idempotent={verification['idempotent_second_run']['is_idempotent']}")
    print(f"  cache_generation_bumped={verification['cache_generation_bumped']}")
    print(f"  private_isolation_ok={verification['private_isolation_ok']}")
    print(f"  shared_retrieval_excludes_needs_review={verification['shared_retrieval_excludes_needs_review']['ok']}")

    await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
