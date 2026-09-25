"""Recover interrupted publication while indexing writers are stopped."""
from typing import Any

from app.rag.vector_store import MongoVectorStore
from app.repositories.documents import DocumentRepository
from app.repositories.versioning import DocumentVersionRepository


async def recover_interrupted_reindexes(*, apply: bool = False, writers_stopped: bool = False) -> list[dict[str, Any]]:
    if apply and not writers_stopped:
        raise ValueError("Stop all indexing writers before applying reindex recovery.")
    versions = DocumentVersionRepository()
    documents = DocumentRepository()
    vectors = MongoVectorStore()
    outcomes = []
    for row in await versions.find_stale_staging():
        outcome = {"version_id": str(row["_id"]), "document_id": row.get("document_id"), "status": "pending"}
        outcomes.append(outcome)
        if not apply:
            continue
        new_id = row.get("document_id")
        if not new_id:
            outcome["status"] = "needs_review_missing_document_id"
            continue
        old = await versions.find_by_id(row["old_version_id"]) if row.get("old_version_id") else None
        if row.get("old_version_id") and (not old or not old.get("document_id")):
            outcome["status"] = "needs_review_missing_previous_version"
            continue
        if old:
            # Never destroy the replacement unless the previous evidence exists.
            if not await vectors.count_by_document_id(old["document_id"]):
                outcome["status"] = "needs_review_missing_previous_chunks"
                continue
            await vectors.activate_version(old["document_id"], new_id)
            await versions.update_by_id(str(old["_id"]), {"document_status": "active", "new_version_id": None})
        await vectors.delete_version_chunks(new_id)
        await documents.delete_by_id(new_id)
        # Release the source lock only after all recovery writes succeeded.
        await versions.delete_by_id(str(row["_id"]))
        outcome["status"] = "recovered_retry_source"
    return outcomes
