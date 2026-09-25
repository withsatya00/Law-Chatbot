"""Deterministic reconciliation between MongoDB and the BM25 index.

MongoDB is the single source of truth. `embeddings_metadata` holds every
retrievable chunk with its ownership metadata; `app/rag/bm25_index.py` keeps a
pickled lexical copy on disk for the sparse retrieval leg. The two are written
by different paths and can drift.

**Observed drift and its cause.** At Phase 2 baseline the BM25 index held 5598
chunks against Mongo's 2079. `BM25Index.add_or_update_chunks` only ever adds,
and the sole removal path (`remove_by_source`) keys on `source_document`.
Re-indexing a document mints fresh `uuid4` chunk ids, so unless the source was
explicitly de-indexed first, the previous generation's ids stay in the pickle
forever. Nothing ever removes a chunk id that no longer exists in Mongo.

That is not cosmetic. A stale BM25 entry is a retrievable copy of text that has
been superseded or deleted, and it carries the ownership metadata it had *at
the time it was written* -- so a stale entry for a private upload can surface
that upload's text after the document was removed.

**What this module will and will not do.**

  * Dry-run is the default everywhere. `analyze()` never writes.
  * `apply()` prunes BM25 entries whose chunk id is absent from Mongo, and
    nothing else. It never touches Mongo, never deletes an uploaded file, and
    never rewrites ownership metadata.
  * It is idempotent: running it twice produces the same index, and the second
    run reports zero stale entries.
  * Chunks that exist in Mongo but not in BM25 are *reported*, not silently
    back-filled -- adding them means re-tokenizing text this module has not
    verified, which is `BM25Index.rebuild()`'s job and an explicit decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from app.models.collections import EMBEDDINGS_METADATA
from app.rag.bm25_index import bm25_index

log = structlog.get_logger(__name__)


@dataclass
class ReconciliationReport:
    """What the two stores disagree about, and what was done about it."""

    mode: str
    mongo_chunk_count: int = 0
    bm25_chunk_count: int = 0
    # In BM25 but not in Mongo. These are the entries `apply()` prunes: text
    # that is still retrievable but no longer backed by the source of truth.
    stale_bm25_chunk_ids: list[str] = field(default_factory=list)
    # In Mongo but not in BM25. Reported only -- see the module docstring.
    missing_from_bm25_count: int = 0
    # Chunk ids appearing more than once inside the BM25 corpus itself.
    duplicate_bm25_chunk_ids: list[str] = field(default_factory=list)
    # Mongo chunks whose `document_id` matches no known document, and chunks
    # whose ownership metadata is internally inconsistent.
    orphaned_source_documents: list[str] = field(default_factory=list)
    ownership_metadata_problems: list[dict[str, Any]] = field(default_factory=list)
    # Stale entries that carried owner metadata -- called out separately
    # because these are the ones with a privacy consequence, not merely a
    # relevance one.
    stale_private_chunk_count: int = 0
    bm25_chunk_count_after: int | None = None
    pruned_count: int = 0

    @property
    def drifted(self) -> bool:
        return bool(
            self.stale_bm25_chunk_ids
            or self.missing_from_bm25_count
            or self.duplicate_bm25_chunk_ids
            or self.ownership_metadata_problems
        )

    def as_dict(self) -> dict[str, Any]:
        """Report shape. Chunk ids are opaque uuids and carry no document text,
        so they are safe to log; the text itself never appears here."""
        return {
            "mode": self.mode,
            "mongo_chunk_count": self.mongo_chunk_count,
            "bm25_chunk_count": self.bm25_chunk_count,
            "bm25_chunk_count_after": self.bm25_chunk_count_after,
            "stale_bm25_records": len(self.stale_bm25_chunk_ids),
            "stale_private_records": self.stale_private_chunk_count,
            "missing_from_bm25": self.missing_from_bm25_count,
            "duplicate_chunk_ids": len(self.duplicate_bm25_chunk_ids),
            "orphaned_source_documents": self.orphaned_source_documents,
            "ownership_metadata_problems": len(self.ownership_metadata_problems),
            "pruned": self.pruned_count,
            "drifted": self.drifted,
        }


class IndexReconciler:
    """Compares Mongo against BM25 and, on request, prunes stale BM25 entries."""

    def __init__(self, index: Any | None = None) -> None:
        # Injectable so tests can drive a real `BM25Index` over a fake corpus
        # without touching the process-wide singleton or its on-disk pickle.
        self.index = index if index is not None else bm25_index

    async def analyze(self) -> ReconciliationReport:
        """Read-only. Never writes to Mongo, BM25 or disk."""
        from app.database.mongodb import mongodb

        report = ReconciliationReport(mode="dry-run")
        collection = mongodb.db[EMBEDDINGS_METADATA]

        mongo_ids: set[str] = set()
        seen_documents: set[str] = set()
        async for row in collection.find({}, {"_id": 1, "document_id": 1, "metadata": 1}):
            chunk_id = str(row["_id"])
            mongo_ids.add(chunk_id)
            metadata = row.get("metadata") or {}
            source = metadata.get("source_document")
            if source:
                seen_documents.add(str(source))
            problem = _ownership_problem(chunk_id, metadata)
            if problem is not None:
                report.ownership_metadata_problems.append(problem)
        report.mongo_chunk_count = len(mongo_ids)

        await self.index.ensure_loaded()
        bm25_ids: list[str] = list(getattr(self.index, "_chunk_ids", []))
        bm25_metadatas: list[dict[str, Any]] = list(getattr(self.index, "_metadatas", []))
        report.bm25_chunk_count = len(bm25_ids)

        seen: set[str] = set()
        for chunk_id in bm25_ids:
            if chunk_id in seen:
                report.duplicate_bm25_chunk_ids.append(chunk_id)
            seen.add(chunk_id)

        for chunk_id, metadata in zip(bm25_ids, bm25_metadatas, strict=False):
            if chunk_id in mongo_ids:
                continue
            report.stale_bm25_chunk_ids.append(chunk_id)
            # A stale entry that was owner-scoped is the privacy-relevant case:
            # it can still be retrieved for its owner after the document is
            # gone, and its filters were frozen at write time.
            if (metadata or {}).get("owner_user_id"):
                report.stale_private_chunk_count += 1

        report.missing_from_bm25_count = len(mongo_ids - set(bm25_ids))
        report.orphaned_source_documents = sorted(
            {
                str((metadata or {}).get("source_document"))
                for chunk_id, metadata in zip(bm25_ids, bm25_metadatas, strict=False)
                if chunk_id not in mongo_ids and (metadata or {}).get("source_document")
            }
            - seen_documents
        )
        return report

    async def apply(self) -> ReconciliationReport:
        """Prune BM25 entries whose chunk id is absent from Mongo.

        Nothing else is written. In particular, chunks present in Mongo but
        missing from BM25 are left alone: adding them back means tokenizing
        text this method has not read, which belongs to an explicit
        `BM25Index.rebuild()`.
        """
        report = await self.analyze()
        report.mode = "apply"
        if not report.stale_bm25_chunk_ids and not report.duplicate_bm25_chunk_ids:
            report.bm25_chunk_count_after = report.bm25_chunk_count
            return report

        stale = set(report.stale_bm25_chunk_ids)
        pruned = await self.index.remove_chunk_ids(stale)
        report.pruned_count = pruned
        report.bm25_chunk_count_after = self.index.chunk_count
        log.info(
            "bm25_reconciliation_applied",
            pruned=pruned,
            stale_private_records=report.stale_private_chunk_count,
            bm25_before=report.bm25_chunk_count,
            bm25_after=report.bm25_chunk_count_after,
        )
        return report


def _ownership_problem(chunk_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Ownership metadata that would misroute a chunk at retrieval time.

    Two shapes matter, and both are reported rather than repaired -- deciding
    that an ambiguous chunk is public, or assigning it an owner, is a
    visibility decision no automated pass should make:

      * `visibility="private"` with no `owner_user_id` -- filtered to nobody,
        or worse, to everybody depending on how a filter is written;
      * `owner_user_id` present but `visibility` says shared -- a private
        upload sitting in the shared knowledge base.
    """
    owner = metadata.get("owner_user_id")
    visibility = metadata.get("visibility")
    if visibility == "private" and not owner:
        return {"chunk_id": chunk_id, "problem": "private_without_owner"}
    if owner and visibility in {"public", "shared", "knowledge_base"}:
        return {"chunk_id": chunk_id, "problem": "owned_chunk_marked_shared"}
    return None
