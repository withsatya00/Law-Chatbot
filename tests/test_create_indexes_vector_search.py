"""Regression coverage for `scripts.create_indexes.create_vector_search_index`'s
Atlas Vector Search index DEFINITION -- specifically, that every field
`MongoVectorStore` unconditionally (or near-unconditionally) folds into a
`$vectorSearch` filter is declared `type: "filter"` here.

Confirmed live 2026-09-23 against a real Atlas Search engine
(`mongodb/mongodb-atlas-local`, see `scripts/_atlas_readiness_verification_
20260923.py`): the index definition was missing `metadata.document_status`,
which `MongoVectorStore._document_status_filter` injects into the filter on
EVERY single call, no exceptions -- Atlas rejected every `$vectorSearch`
aggregation with "Path 'metadata.document_status' needs to be indexed as
filter", and `MongoVectorStore.search` caught that as a `PyMongoError` and
silently fell back to the slow local scan. An Atlas-backed deployment
missing this field would never actually use `$vectorSearch` at all, on any
request, ever -- worse than the earlier Phase 4A gap (owner_session_id/
owner_user_id/section_number), which only affected ownership/citation
queries specifically. This test catches a regression of either gap without
needing a live Atlas deployment to run in CI.
"""

from app.core.config import settings
from app.database.mongodb import mongodb
from scripts.create_indexes import create_vector_search_index


class _FakeCollection:
    def __init__(self) -> None:
        self.captured_definition: dict | None = None

    async def create_search_index(self, definition: dict) -> None:
        self.captured_definition = definition


class _FakeDB:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def __getitem__(self, _name: str) -> _FakeCollection:
        return self._collection


class _FakeClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self._db = _FakeDB(collection)

    def __getitem__(self, _name: str) -> _FakeDB:
        return self._db


async def _filter_field_paths(monkeypatch) -> set[str]:
    fake_collection = _FakeCollection()
    monkeypatch.setattr(mongodb, "_client", _FakeClient(fake_collection))
    await create_vector_search_index()
    assert fake_collection.captured_definition is not None
    fields = fake_collection.captured_definition["definition"]["fields"]
    return {f["path"] for f in fields if f["type"] == "filter"}


async def test_index_declares_document_status_as_filterable(monkeypatch) -> None:
    """`MongoVectorStore._document_status_filter` adds this to EVERY search
    call's filter, unconditionally -- see this test module's docstring."""
    paths = await _filter_field_paths(monkeypatch)
    assert "metadata.document_status" in paths


async def test_index_declares_every_part45_46_ownership_field_as_filterable(monkeypatch) -> None:
    """Phase 4A "Atlas Vector Filter Repair" regression: `owner_session_id`/
    `owner_user_id` are in the `$or` branches `ChatService._prepare_rag_
    context` builds on every call, and `section_number` is added for every
    SECTION_LOOKUP/citation-style query."""
    paths = await _filter_field_paths(monkeypatch)
    assert {"metadata.owner_session_id", "metadata.owner_user_id", "metadata.section_number"} <= paths


async def test_index_vector_field_dimensions_match_configured_embedding_model(monkeypatch) -> None:
    fake_collection = _FakeCollection()
    monkeypatch.setattr(mongodb, "_client", _FakeClient(fake_collection))
    await create_vector_search_index()
    fields = fake_collection.captured_definition["definition"]["fields"]
    vector_field = next(f for f in fields if f["type"] == "vector")
    assert vector_field["path"] == "embedding"
    assert vector_field["numDimensions"] == settings.embedding_dimensions
    assert vector_field["similarity"] == "cosine"
