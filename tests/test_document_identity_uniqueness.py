"""Regression tests for security finding C4 (duplicate document identity).

Root cause: nothing in the schema stopped the same logical document/version
(the same `document_hash` + `version_number`) from being written twice under
different `source_document` names -- `IndexingPipeline.index_file`'s
`_known_hashes()` pre-check is an application-level guard against a NEW
duplicate for any write that goes through it, not a database constraint, so
a bug or a write path that bypassed it could still leave two `active` rows
disagreeing about the same document's identity and review state, with
retrieval and admin review having no way to know they were ever the same
document.

Fixed with a partial unique index (`uniq_shared_document_identity` on
`document_versions`, `{document_hash, version_number}`, scoped to shared,
currently-`active` rows) plus `DocumentVersionRepository.
find_duplicate_shared_identities` to report any pre-existing violations for
an admin to reconcile (this deployment's own data had exactly one, an
active+superseded historical pair, deliberately out of this constraint's
scope -- see that method's docstring).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from uuid import uuid4

import pytest
from pymongo import ASCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

from app.database.mongodb import mongodb
from app.models.collections import DOCUMENT_VERSIONS
from app.repositories.versioning import DocumentVersionRepository

T = TypeVar("T")


@pytest.fixture(scope="module")
def loop():
    ev_loop = asyncio.new_event_loop()
    # Motor's driver looks up "the current loop" internally on first real
    # use, not only via whatever loop `run_until_complete` happens to be
    # executing on -- without this, it can bind to a stale/default loop
    # instead of `ev_loop`, and every query then fails with 'future belongs
    # to a different loop'.
    asyncio.set_event_loop(ev_loop)
    mongodb._client = None
    ev_loop.run_until_complete(mongodb.connect())
    yield ev_loop
    ev_loop.run_until_complete(mongodb.close())
    ev_loop.close()
    asyncio.set_event_loop(None)


def run[T](loop, coro: Coroutine[Any, Any, T]) -> T:
    return loop.run_until_complete(coro)


def _row(**overrides: Any) -> dict[str, Any]:
    base = {
        "_id": str(uuid4()),
        "source_document": f"doc-{uuid4()}.pdf",
        "owner_user_id": None,
        "owner_session_id": None,
        "document_status": "active",
        "document_hash": f"hash-{uuid4()}",
        "version_number": 1,
    }
    base.update(overrides)
    return base


def test_the_unique_index_exists_and_is_scoped_to_shared_active_rows(loop) -> None:
    indexes = run(loop, mongodb.db[DOCUMENT_VERSIONS].index_information())
    assert "uniq_shared_document_identity" in indexes
    spec = indexes["uniq_shared_document_identity"]
    assert spec["unique"] is True
    assert spec["partialFilterExpression"] == {
        "owner_user_id": None, "owner_session_id": None, "document_status": "active",
    }


def test_a_second_active_row_with_the_same_identity_is_rejected_at_the_db_level(loop) -> None:
    """The core C4 scenario: the same content, indexed twice under
    different filenames, both ending up `active`."""
    document_hash = f"c4-dup-{uuid4()}"
    first = _row(document_hash=document_hash, version_number=1)
    second = _row(document_hash=document_hash, version_number=1)
    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(first))

    with pytest.raises(DuplicateKeyError):
        run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(second))

    # Independent re-fetch: only the first row landed.
    matches = run(loop, mongodb.db[DOCUMENT_VERSIONS].count_documents({"document_hash": document_hash}))
    assert matches == 1

    run(loop, mongodb.db[DOCUMENT_VERSIONS].delete_one({"_id": first["_id"]}))


def test_a_superseded_row_does_not_block_a_new_active_row_with_the_same_identity(loop) -> None:
    """Version-lifecycle history (an old, no-longer-serving row) sharing an
    identity with the current active one is not a live contradiction and
    must not be constrained the same way as two competing active rows."""
    document_hash = f"c4-lifecycle-{uuid4()}"
    old = _row(document_hash=document_hash, version_number=1, document_status="superseded")
    new = _row(document_hash=document_hash, version_number=1, document_status="active")
    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(old))

    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(new))  # must not raise

    matches = run(loop, mongodb.db[DOCUMENT_VERSIONS].count_documents({"document_hash": document_hash}))
    assert matches == 2

    run(loop, mongodb.db[DOCUMENT_VERSIONS].delete_many({"document_hash": document_hash}))


def test_two_different_private_owners_may_share_an_identity(loop) -> None:
    """Two different users privately uploading byte-identical files is
    expected and must keep working -- the constraint is shared-KB only."""
    document_hash = f"c4-private-{uuid4()}"
    first = _row(document_hash=document_hash, version_number=1, owner_user_id="user-a", owner_session_id=None)
    second = _row(document_hash=document_hash, version_number=1, owner_user_id="user-b", owner_session_id=None)
    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(first))

    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(second))  # must not raise

    matches = run(loop, mongodb.db[DOCUMENT_VERSIONS].count_documents({"document_hash": document_hash}))
    assert matches == 2

    run(loop, mongodb.db[DOCUMENT_VERSIONS].delete_many({"document_hash": document_hash}))


def test_find_duplicate_shared_identities_would_report_an_active_vs_active_duplicate(loop) -> None:
    """The index (tested above) makes a genuine active/active duplicate
    unreachable through normal inserts -- that IS the fix. This drops the
    index temporarily to prove the detector's aggregation itself is
    correct, independent of the constraint that now makes its target state
    unreachable in production; the index is always recreated before the
    assertion, in a `finally`, so this test cannot leave it missing for any
    other test in this module or a real deployment."""
    document_hash = f"c4-report-{uuid4()}"
    row_a = _row(document_hash=document_hash, version_number=1, source_document="a.pdf")
    row_b = _row(document_hash=document_hash, version_number=1, source_document="b.pdf")

    async def _seed_without_index_and_report() -> list[dict[str, Any]]:
        await mongodb.db[DOCUMENT_VERSIONS].drop_index("uniq_shared_document_identity")
        try:
            await mongodb.db[DOCUMENT_VERSIONS].insert_one(row_a)
            await mongodb.db[DOCUMENT_VERSIONS].insert_one(row_b)
            return await DocumentVersionRepository().find_duplicate_shared_identities()
        finally:
            await mongodb.db[DOCUMENT_VERSIONS].delete_many({"document_hash": document_hash})
            await mongodb.db[DOCUMENT_VERSIONS].create_indexes([
                IndexModel(
                    [("document_hash", ASCENDING), ("version_number", ASCENDING)],
                    unique=True,
                    partialFilterExpression={
                        "owner_user_id": None, "owner_session_id": None, "document_status": "active",
                    },
                    name="uniq_shared_document_identity",
                )
            ])

    duplicates = run(loop, _seed_without_index_and_report())

    matching = [group for group in duplicates if group["_id"]["document_hash"] == document_hash]
    assert len(matching) == 1
    assert matching[0]["count"] == 2
    names = {record["source_document"] for record in matching[0]["records"]}
    assert names == {"a.pdf", "b.pdf"}

    # And the index is back in place for every subsequent test/deployment use.
    indexes = run(loop, mongodb.db[DOCUMENT_VERSIONS].index_information())
    assert "uniq_shared_document_identity" in indexes


def test_find_duplicate_shared_identities_ignores_a_harmless_active_superseded_pair(loop) -> None:
    document_hash = f"c4-harmless-{uuid4()}"
    old = _row(document_hash=document_hash, version_number=1, document_status="superseded")
    new = _row(document_hash=document_hash, version_number=1, document_status="active")
    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(old))
    run(loop, mongodb.db[DOCUMENT_VERSIONS].insert_one(new))

    duplicates = run(loop, DocumentVersionRepository().find_duplicate_shared_identities())

    run(loop, mongodb.db[DOCUMENT_VERSIONS].delete_many({"document_hash": document_hash}))

    assert not any(group["_id"]["document_hash"] == document_hash for group in duplicates)
