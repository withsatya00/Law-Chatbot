"""Structural coverage for `MongoVectorStore._atlas_filter`/`_atlas_vector_leg`.

This dev environment has no Atlas backend (no `$vectorSearch`-capable Mongo
deployment) -- every prior session confirmed retrieval always falls back to
`_local_cosine_leg` here, so the Atlas code path has never been exercised
against live traffic. A real end-to-end test isn't possible without an
actual Atlas cluster, but the query-BUILDING logic (`_atlas_filter`'s
translation of this app's filter shapes into Atlas `$vectorSearch` filter
syntax, and `_atlas_vector_leg`'s aggregation pipeline construction /
result parsing) is pure and fully testable without one. These tests catch
the class of bug that would otherwise only surface the first time this code
runs against a real Atlas cluster: a wrong operator, a wrong field path, or
a filter shape Atlas's `$vectorSearch` syntax doesn't actually accept.
"""

import pytest

from app.database.mongodb import mongodb
from app.rag.vector_store import MongoVectorStore


def test_atlas_filter_scalar_becomes_eq() -> None:
    store = MongoVectorStore()
    result = store._atlas_filter({"owner_user_id": "user-1"})
    assert result == {"metadata.owner_user_id": {"$eq": "user-1"}}


def test_atlas_filter_list_without_null_becomes_in() -> None:
    store = MongoVectorStore()
    result = store._atlas_filter({"act_name": ["The Bharatiya Nyaya Sanhita", "The Indian Penal Code"]})
    assert result == {"metadata.act_name": {"$in": ["The Bharatiya Nyaya Sanhita", "The Indian Penal Code"]}}


def test_atlas_filter_list_with_null_becomes_or_of_eq_null_and_in() -> None:
    # Confirmed live (2026-08-24) against a real Atlas Search deployment
    # (mongodb/mongodb-atlas-local): a literal `null` INSIDE `$in` is
    # rejected outright by mongot ("value type cannot be null") -- see
    # `MongoVectorStore._atlas_filter`'s docstring for the full story and
    # the verification script. `$eq: null` is the operator Atlas actually
    # accepts for this app's explicit-null-valued fields.
    store = MongoVectorStore()
    result = store._atlas_filter({"owner_session_id": [None, "session-1"]})
    assert result == {
        "$or": [
            {"metadata.owner_session_id": {"$eq": None}},
            {"metadata.owner_session_id": {"$in": ["session-1"]}},
        ]
    }


def test_atlas_filter_list_of_only_null_becomes_bare_eq_null() -> None:
    store = MongoVectorStore()
    result = store._atlas_filter({"owner_user_id": [None]})
    assert result == {"metadata.owner_user_id": {"$eq": None}}


def test_atlas_filter_empty_values_are_dropped() -> None:
    store = MongoVectorStore()
    result = store._atlas_filter({"owner_user_id": None, "act_name": []})
    assert result is None


def test_atlas_filter_or_matches_part46_ownership_shape() -> None:
    """Mirrors the exact filter `ChatService._prepare_rag_context` builds:
    global-or-mine-by-session, or mine-by-authenticated-user.
    """
    store = MongoVectorStore()
    filters = {
        "$or": [
            {"owner_session_id": [None, "session-1"], "owner_user_id": [None]},
            {"owner_user_id": ["user-1"]},
        ]
    }
    result = store._atlas_filter(filters)
    assert result == {
        "$or": [
            {"$and": [
                {"$or": [
                    {"metadata.owner_session_id": {"$eq": None}},
                    {"metadata.owner_session_id": {"$in": ["session-1"]}},
                ]},
                {"metadata.owner_user_id": {"$eq": None}},
            ]},
            {"metadata.owner_user_id": {"$in": ["user-1"]}},
        ]
    }


def test_atlas_filter_combines_or_with_sibling_equality() -> None:
    store = MongoVectorStore()
    filters = {"act_name": ["The Bharatiya Nyaya Sanhita"], "$or": [{"owner_user_id": ["user-1"]}]}
    result = store._atlas_filter(filters)
    assert result == {
        "$and": [
            {"$or": [{"metadata.owner_user_id": {"$in": ["user-1"]}}]},
            {"metadata.act_name": {"$in": ["The Bharatiya Nyaya Sanhita"]}},
        ]
    }


class _FakeCursor:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self._items:
            yield item


class _FakeCollection:
    def __init__(self, items: list[dict]) -> None:
        self._items = items
        self.captured_pipeline: list[dict] | None = None

    def aggregate(self, pipeline: list[dict]) -> _FakeCursor:
        self.captured_pipeline = pipeline
        return _FakeCursor(self._items)


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


@pytest.mark.asyncio
async def test_atlas_vector_leg_builds_valid_vector_search_stage_and_parses_results(monkeypatch) -> None:
    fake_collection = _FakeCollection(
        [{"_id": "chunk-1", "text": "Section 2 text", "metadata": {"act_name": "The Bharatiya Nyaya Sanhita"}, "score": 0.91}]
    )
    monkeypatch.setattr(mongodb, "_client", _FakeClient(fake_collection))

    store = MongoVectorStore()
    results = await store._atlas_vector_leg([0.1, 0.2, 0.3], limit=5, filters={"owner_user_id": "user-1"})

    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == pytest.approx(0.91)
    assert results[0].metadata["act_name"] == "The Bharatiya Nyaya Sanhita"

    assert fake_collection.captured_pipeline is not None
    vector_stage = fake_collection.captured_pipeline[0]["$vectorSearch"]
    assert vector_stage["path"] == "embedding"
    assert vector_stage["queryVector"] == [0.1, 0.2, 0.3]
    assert vector_stage["limit"] == 5
    assert vector_stage["numCandidates"] >= 5
    assert vector_stage["filter"] == {"metadata.owner_user_id": {"$eq": "user-1"}}
    assert fake_collection.captured_pipeline[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}


@pytest.mark.asyncio
async def test_atlas_vector_leg_omits_filter_key_entirely_when_no_filters(monkeypatch) -> None:
    fake_collection = _FakeCollection([])
    monkeypatch.setattr(mongodb, "_client", _FakeClient(fake_collection))

    store = MongoVectorStore()
    await store._atlas_vector_leg([0.1, 0.2], limit=3, filters={})

    vector_stage = fake_collection.captured_pipeline[0]["$vectorSearch"]
    assert "filter" not in vector_stage
