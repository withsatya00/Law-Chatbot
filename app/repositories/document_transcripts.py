from datetime import UTC, datetime
from typing import Any

from app.models.collections import DOCUMENT_TRANSCRIPTS
from app.repositories.base import MongoRepository


class DocumentTranscriptRepository(MongoRepository):
    """The most recent professionally-formatted retype of one uploaded
    document, keyed by `document_id` (one row per document, not one per
    transcription request) so a later export re-renders this stored text
    instead of paying for another LLM call per download.
    """

    collection_name = DOCUMENT_TRANSCRIPTS

    async def save(
        self, document_id: str, owner_user_id: str | None, owner_session_id: str | None, data: dict[str, Any],
    ) -> None:
        await self.collection.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "owner_user_id": owner_user_id,
                    "owner_session_id": owner_session_id,
                    "data": data,
                    "updated_at": datetime.now(UTC),
                },
                "$setOnInsert": {"created_at": datetime.now(UTC)},
            },
            upsert=True,
        )

    async def find_by_document_id(self, document_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": document_id})
