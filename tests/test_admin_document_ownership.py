"""`GET /admin/documents/unowned` / `POST /admin/documents/{doc}/assign-owner`
(see [[project_gap_closure_round1]]) had zero test coverage when built --
closing that flagged gap. Calls the route functions directly (same pattern
as `tests/test_auth_routes.py`), repository methods mocked with `AsyncMock`
(same pattern as `tests/test_kb_ingestion_service.py`) rather than hitting
a real MongoDB.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.api import admin
from app.core.exceptions import NotFoundError
from app.repositories.documents import EmbeddingMetadataRepository
from app.schemas.admin import AssignDocumentOwnershipRequest


def test_list_unowned_documents_returns_repository_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        EmbeddingMetadataRepository, "list_unowned_documents",
        AsyncMock(return_value=[{"source_document": "Old_Upload.pdf", "chunk_count": 12}]),
    )

    result = asyncio.run(admin.list_unowned_documents())

    assert result["count"] == 1
    assert result["documents"][0]["source_document"] == "Old_Upload.pdf"


def test_list_unowned_documents_empty_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EmbeddingMetadataRepository, "list_unowned_documents", AsyncMock(return_value=[]))

    result = asyncio.run(admin.list_unowned_documents())

    assert result == {"count": 0, "documents": []}


def test_assign_document_ownership_with_owner_id(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = AsyncMock(return_value=3)
    monkeypatch.setattr(EmbeddingMetadataRepository, "assign_document_ownership", mock)

    result = asyncio.run(
        admin.assign_document_ownership("Old_Upload.pdf", AssignDocumentOwnershipRequest(owner_user_id="user-1"))
    )

    assert result == {"source_document": "Old_Upload.pdf", "chunks_updated": 3, "owner_user_id": "user-1"}
    mock.assert_awaited_once_with("Old_Upload.pdf", "user-1")


def test_assign_document_ownership_without_owner_id_marks_reviewed_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Confirming "stays global" (no owner_user_id) is a distinct, valid
    # decision -- not treated as a no-op or an error.
    mock = AsyncMock(return_value=5)
    monkeypatch.setattr(EmbeddingMetadataRepository, "assign_document_ownership", mock)

    result = asyncio.run(
        admin.assign_document_ownership("Curated_Act.pdf", AssignDocumentOwnershipRequest())
    )

    assert result["chunks_updated"] == 5
    assert result["owner_user_id"] is None
    mock.assert_awaited_once_with("Curated_Act.pdf", None)


def test_assign_document_ownership_raises_not_found_for_unknown_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EmbeddingMetadataRepository, "assign_document_ownership", AsyncMock(return_value=0))

    with pytest.raises(NotFoundError):
        asyncio.run(
            admin.assign_document_ownership("Does_Not_Exist.pdf", AssignDocumentOwnershipRequest())
        )
