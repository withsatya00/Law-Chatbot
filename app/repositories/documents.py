from typing import Any

from app.models.collections import EMBEDDINGS_METADATA, UPLOADED_DOCUMENTS
from app.repositories.base import MongoRepository


class DocumentRepository(MongoRepository):
    collection_name = UPLOADED_DOCUMENTS


class EmbeddingMetadataRepository(MongoRepository):
    collection_name = EMBEDDINGS_METADATA

    async def upsert_chunk(self, chunk_id: str, payload: dict[str, Any]) -> None:
        await self.collection.update_one({"_id": chunk_id}, {"$set": payload}, upsert=True)

    async def list_unowned_documents(self) -> list[dict[str, Any]]:
        """Part 45/46 stamp `owner_session_id`/`owner_user_id` at ingest, but
        documents indexed BEFORE those fixes shipped (flagged, never fixed,
        in [[project_part45_per_user_document_isolation]]) have neither
        field at all -- structurally indistinguishable from a deliberately
        global/curated document, since both cases just have the field
        absent. `metadata.ownership_reviewed` (new) is how an admin marks
        "yes, I looked at this one and it's meant to stay global" so it
        stops showing up here after review, without needing to fabricate an
        owner that was never actually recorded.
        """
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "metadata.owner_session_id": {"$exists": False},
                    "metadata.owner_user_id": {"$exists": False},
                    "metadata.ownership_reviewed": {"$ne": True},
                }
            },
            {
                "$group": {
                    "_id": "$metadata.source_document",
                    "chunk_count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        return [
            {"source_document": doc["_id"], "chunk_count": doc["chunk_count"]}
            async for doc in self.collection.aggregate(pipeline)
        ]

    async def assign_document_ownership(self, source_document: str, owner_user_id: str | None) -> int:
        """Bulk-updates every chunk of one source document. Always sets
        `ownership_reviewed` (an admin looked at this document either way),
        and additionally sets `owner_user_id` when the admin supplied one --
        e.g. an admin who knows, from outside this system, who actually
        uploaded it. Omitting `owner_user_id` is an explicit, recorded
        decision that this document stays globally visible, replacing the
        previous SILENT default of "absent field = accidentally global
        forever."
        """
        updates: dict[str, Any] = {"metadata.ownership_reviewed": True}
        if owner_user_id:
            updates["metadata.owner_user_id"] = owner_user_id
        result = await self.collection.update_many({"metadata.source_document": source_document}, {"$set": updates})
        return result.modified_count
