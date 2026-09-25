"""Repositories for the E-Notarization module.

`NotarizationAuditRepository` is intentionally NOT a `MongoRepository`
subclass: inheriting would hand every caller `update_by_id` and
`delete_by_id`, which is precisely what an immutable audit trail must not
offer. It composes the collection instead and exposes append + read only.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection

from app.database.mongodb import mongodb
from app.models.collections import (
    ESIGN_SESSIONS,
    NOTARIZATION_AUDIT_EVENTS,
    NOTARIZATION_DOCUMENTS,
    NOTARIZATION_REQUESTS,
    NOTARY_ACCOUNTS,
)
from app.repositories.base import MongoRepository


class NotarizationDocumentRepository(MongoRepository):
    collection_name = NOTARIZATION_DOCUMENTS

    async def latest_version(self, source_draft_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"source_draft_id": source_draft_id}, sort=[("document_version", -1)]
        )

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_user_id": user_id}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def find_by_verification_token(self, token: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"verification_token": token})


class NotaryAccountRepository(MongoRepository):
    collection_name = NOTARY_ACCOUNTS

    async def find_by_user_id(self, user_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"user_id": user_id})

    async def find_by_registration_number(self, registration_number: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"registration_number": registration_number})

    async def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find({}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)


class NotarizationRequestRepository(MongoRepository):
    collection_name = NOTARIZATION_REQUESTS

    async def list_for_notary(self, notary_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"assigned_notary_id": notary_id}
        if status:
            query["review_status"] = status
        cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.collection.find({"review_status": "pending"}).sort("created_at", 1).limit(limit)
        return await cursor.to_list(length=limit)

    async def list_for_requester(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.collection.find({"requested_by_user_id": user_id}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def find_for_document(self, document_id: str) -> list[dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", -1)
        return await cursor.to_list(length=100)


class ESignSessionRepository(MongoRepository):
    collection_name = ESIGN_SESSIONS

    async def find_by_provider_reference(self, provider: str, provider_reference: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"provider": provider, "provider_reference": provider_reference})

    async def list_for_document(self, document_id: str, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def list_failed(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = (
            self.collection.find({"status": {"$in": ["failed", "expired", "cancelled"]}})
            .sort("created_at", -1)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)


class NotarizationAuditRepository:
    """Append-only store for notarization audit events.

    Exposes no update or delete. This is the application-level half of the
    immutability guarantee; the operator-level half (an append-only database
    role) is documented in docs/NOTARIZATION.md.
    """

    collection_name = NOTARIZATION_AUDIT_EVENTS

    @property
    def collection(self) -> AsyncIOMotorCollection[Any]:
        return mongodb.db[self.collection_name]

    async def append(self, event: dict[str, Any]) -> str:
        event.setdefault("_id", str(uuid4()))
        event.setdefault("occurred_at", datetime.now(UTC))
        # `recorded_at` is OUR clock at write time, distinct from
        # `occurred_at` (when the action happened). Divergence between the
        # two is itself evidence worth keeping.
        event["recorded_at"] = datetime.now(UTC)
        await self.collection.insert_one(event)
        return str(event["_id"])

    async def list_for_document(self, document_id: str, limit: int = 200) -> list[dict[str, Any]]:
        cursor = self.collection.find({"document_id": document_id}).sort("occurred_at", 1).limit(limit)
        return await cursor.to_list(length=limit)

    async def list_recent(self, limit: int = 200, action: str | None = None) -> list[dict[str, Any]]:
        query = {"action": action} if action else {}
        cursor = self.collection.find(query).sort("occurred_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
