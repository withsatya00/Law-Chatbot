from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, *_args):
        return self

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield row
        return iterate()


def test_root_returns_api_discovery_instead_of_404() -> None:
    from app.main import create_app

    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"
    assert response.json()["docs"] == "/docs"


def test_internal_metrics_is_reachable_unauthenticated_in_prometheus_format() -> None:
    """Deliberately unauthenticated -- see the route's own docstring in
    app/main.py for why (a Prometheus scraper hitting this on every interval
    can't practically carry a short-lived admin JWT); trust boundary is the
    deployment's network, not an app-level check.
    """
    from app.main import create_app
    from app.observability.metrics import metrics

    metrics.increment("http_smoke_test_counter")

    response = TestClient(create_app()).get("/internal/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_smoke_test_counter" in response.text


def test_document_listing_is_owner_scoped() -> None:
    from app.services.document_service import DocumentService

    service = DocumentService()
    # Motor's find is synchronous and returns an async cursor.
    find = lambda query, projection: _Cursor([{
        "_id": "doc-1", "filename": "notice.pdf", "chunk_count": 3,
        "index_status": "indexed",
    }])
    service.documents = SimpleNamespace(collection=SimpleNamespace(find=find))

    result = asyncio.run(service.list_owned(user_id="user-a"))

    assert result == [{
        "document_id": "doc-1", "filename": "notice.pdf", "detected_language": None,
        "chunks_indexed": 3, "status": "indexed", "created_at": None, "updated_at": None,
    }]


def test_document_listing_prefers_the_original_filename_over_the_storage_key() -> None:
    """Security/correctness finding K2: `filename` on the stored document
    record is the server-generated storage key
    (`f"{uuid4()}{suffix}"`), never the caller's own filename -- confirmed
    live: `document_versions` rows exist with `source_document` values like
    "62f33814-076e-4750-833f-2eb3aad43e00.pdf" instead of the file the user
    actually uploaded. `original_filename` (this fix) must be preferred
    when present."""
    from app.services.document_service import DocumentService

    service = DocumentService()
    find = lambda query, projection: _Cursor([{
        "_id": "doc-1", "filename": "a1b2c3d4-5678-90ab-cdef-1234567890ab.pdf",
        "original_filename": "my_rent_agreement.pdf", "chunk_count": 3, "index_status": "indexed",
    }])
    service.documents = SimpleNamespace(collection=SimpleNamespace(find=find))

    result = asyncio.run(service.list_owned(user_id="user-a"))

    assert result[0]["filename"] == "my_rent_agreement.pdf"


def test_document_listing_falls_back_to_the_storage_key_when_no_original_filename_was_recorded() -> None:
    """Backward compatibility: a document indexed before `original_filename`
    existed (or via a caller with no such concept, e.g. KB ingestion) must
    not crash or show a blank name."""
    from app.services.document_service import DocumentService

    service = DocumentService()
    find = lambda query, projection: _Cursor([{
        "_id": "doc-1", "filename": "legacy_document.pdf", "chunk_count": 1, "index_status": "indexed",
    }])
    service.documents = SimpleNamespace(collection=SimpleNamespace(find=find))

    result = asyncio.run(service.list_owned(user_id="user-a"))

    assert result[0]["filename"] == "legacy_document.pdf"


def test_anonymous_listing_without_session_returns_empty_without_db_access() -> None:
    from app.services.document_service import DocumentService

    service = DocumentService()
    service.documents = AsyncMock()
    assert asyncio.run(service.list_owned()) == []


def test_documents_route_is_registered_for_get() -> None:
    from app.api.upload import router

    route = next(route for route in router.routes if route.path == "/documents")
    assert "GET" in route.methods


def test_document_delete_route_is_registered() -> None:
    from app.api.upload import router

    route = next(route for route in router.routes if route.path == "/documents/{document_id}")
    assert "DELETE" in route.methods


def test_claimed_session_cannot_be_listed_anonymously() -> None:
    from app.core.exceptions import ForbiddenError
    from app.services.document_service import DocumentService

    service = DocumentService()
    service.memory.check_access = AsyncMock(side_effect=ForbiddenError("different account"))
    with pytest.raises(ForbiddenError):
        asyncio.run(service.list_owned(session_id="private-session"))
