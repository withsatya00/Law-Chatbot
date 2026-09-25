"""Part 45 "Per-User Document Isolation": `MongoVectorStore._mongo_filter`/
`_atlas_filter` filter-clause builders, tested in isolation from MongoDB
(pure dict-in/dict-out functions, no I/O -- constructing `MongoVectorStore()`
itself doesn't touch the database).
"""

from app.rag.vector_store import MongoVectorStore


def _store() -> MongoVectorStore:
    return MongoVectorStore()


def test_mongo_filter_scalar_value_is_plain_equality_unchanged() -> None:
    assert _store()._mongo_filter({"source_document": "bnss.pdf"}) == {"metadata.source_document": "bnss.pdf"}


def test_mongo_filter_list_value_becomes_in_clause() -> None:
    result = _store()._mongo_filter({"owner_session_id": [None, "session-A"]})
    assert result == {"metadata.owner_session_id": {"$in": [None, "session-A"]}}


def test_mongo_filter_combines_scalar_and_list_clauses() -> None:
    result = _store()._mongo_filter({"source_document": "x.pdf", "owner_session_id": [None, "session-A"]})
    assert result == {
        "metadata.source_document": "x.pdf",
        "metadata.owner_session_id": {"$in": [None, "session-A"]},
    }


def test_mongo_filter_skips_falsy_values() -> None:
    assert _store()._mongo_filter({"source_document": "", "owner_session_id": None}) == {}


def test_atlas_filter_scalar_value_is_eq_unchanged() -> None:
    assert _store()._atlas_filter({"source_document": "bnss.pdf"}) == {"metadata.source_document": {"$eq": "bnss.pdf"}}


def test_atlas_filter_list_value_becomes_in_clause() -> None:
    # A `None` element can't stay in a plain `$in` here -- confirmed live
    # (2026-08-24) against a real Atlas Search deployment that Atlas
    # rejects a literal null inside `$in` outright. See
    # `MongoVectorStore._atlas_filter`'s docstring and
    # [[project_atlas_vectorsearch_live_verification]].
    result = _store()._atlas_filter({"owner_session_id": [None, "session-A"]})
    assert result == {
        "$or": [
            {"metadata.owner_session_id": {"$eq": None}},
            {"metadata.owner_session_id": {"$in": ["session-A"]}},
        ]
    }


def test_atlas_filter_combines_scalar_and_list_clauses_with_and() -> None:
    result = _store()._atlas_filter({"source_document": "x.pdf", "owner_session_id": [None, "session-A"]})
    assert result == {
        "$and": [
            {"metadata.source_document": {"$eq": "x.pdf"}},
            {
                "$or": [
                    {"metadata.owner_session_id": {"$eq": None}},
                    {"metadata.owner_session_id": {"$in": ["session-A"]}},
                ]
            },
        ]
    }


def test_atlas_filter_returns_none_when_no_clauses() -> None:
    assert _store()._atlas_filter({}) is None


# ---------------------------------------------------------------------------
# Part 46 "Authenticated User Ownership": a reserved `"$or"` key expresses
# "global OR mine (by session) OR mine (by authenticated user)" -- a shape
# the flat AND-of-memberships above can't express alone.
# ---------------------------------------------------------------------------

_OWNER_OR_FILTER = {
    "$or": [
        {"owner_session_id": [None, "session-1"], "owner_user_id": [None]},
        {"owner_user_id": ["user-A"]},
    ]
}


def test_mongo_filter_or_recurses_each_branch_with_existing_semantics() -> None:
    result = _store()._mongo_filter(_OWNER_OR_FILTER)
    assert result == {
        "$or": [
            {"metadata.owner_session_id": {"$in": [None, "session-1"]}, "metadata.owner_user_id": {"$in": [None]}},
            {"metadata.owner_user_id": {"$in": ["user-A"]}},
        ]
    }


def test_mongo_filter_or_combines_with_sibling_keys_as_implicit_and() -> None:
    result = _store()._mongo_filter({"source_document": "x.pdf", **_OWNER_OR_FILTER})
    assert result["metadata.source_document"] == "x.pdf"
    assert "$or" in result


def test_atlas_filter_or_recurses_each_branch_with_existing_semantics() -> None:
    result = _store()._atlas_filter(_OWNER_OR_FILTER)
    assert result == {
        "$or": [
            {
                "$and": [
                    {
                        "$or": [
                            {"metadata.owner_session_id": {"$eq": None}},
                            {"metadata.owner_session_id": {"$in": ["session-1"]}},
                        ]
                    },
                    {"metadata.owner_user_id": {"$eq": None}},
                ]
            },
            {"metadata.owner_user_id": {"$in": ["user-A"]}},
        ]
    }
