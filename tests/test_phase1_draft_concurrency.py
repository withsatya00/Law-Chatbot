"""Real-database concurrency coverage for draft edits.

Same opt-in convention and disposable-service pattern as
tests/test_phase1_live_services.py (PHASE1_LIVE_TESTS=1,
docker-compose.phase1-test.yml's Mongo :37017 / Redis :36379): real Mongo,
real Redis. Only the LLM is stubbed (via `LLMFactory.create`/
`create_resilient`), so no paid model budget is needed.

This is the one class of bug a mocked repository structurally cannot catch:
`unittest.mock.AsyncMock` returns are synchronous and deterministic, so two
"concurrent" calls against a mock never actually interleave. Only a real
database connection produces the genuine read-then-write race that
`LegalDraftEngine.regenerate` (app/drafting/engine.py:600) is exposed to --
see QA_TEST_MATRIX_20260911.md, session 4, "BUG-015 / P4-T7 (concurrent
edits)", where this was first observed live against real Gemini/MongoDB.
This file reproduces the same defect deterministically, on demand, against
disposable real infrastructure, rather than relying on it turning up again
during manual live testing.
"""
import asyncio
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.database.mongodb import mongodb
from app.drafting.engine import LegalDraftEngine
from app.llm.base import LLMResponse
from app.llm.factory import LLMFactory
from app.models.collections import LEGAL_DRAFTS
from app.rag.bm25_index import BM25Index
from app.schemas.drafting import DraftGenerateRequest

pytestmark = pytest.mark.skipif(os.environ.get("PHASE1_LIVE_TESTS") != "1", reason="requires isolated Phase 1 services")

_FIELDS = {
    "applicant_name": "Deepak Nair",
    "applicant_address": "7 Lake View Road, Kochi",
    "applicant_mobile": "9988776655",
    "respondent_name": "Coastal Traders Pvt Ltd",
    "respondent_address": "22 Marine Drive, Kochi",
    "facts": "On 3 March 2026 I paid Rs. 45000 in advance for furniture that was never delivered.",
    "expected_relief": "Refund of Rs. 45000 within 15 days.",
    "place": "Kochi",
}


@pytest.fixture
async def isolated_services(monkeypatch, tmp_path):
    """Identical to test_phase1_live_services.py's fixture of the same name --
    duplicated rather than imported, matching this repo's existing convention
    of one integration test file per fixture setup (there is no shared
    tests/conftest.py)."""
    client = AsyncIOMotorClient("mongodb://127.0.0.1:37017", serverSelectionTimeoutMS=3000)
    cache = Redis.from_url("redis://127.0.0.1:36379/0")
    database = "phase1_test_" + uuid4().hex
    monkeypatch.setattr(mongodb, "_client", client)
    monkeypatch.setattr(redis_client, "_client", cache)
    monkeypatch.setattr(settings, "mongodb_database", database)
    monkeypatch.setattr(settings, "vector_search_backend", "local")
    import app.rag.vector_store as vector_module
    index = BM25Index(tmp_path / "bm25.pkl")
    monkeypatch.setattr(vector_module, "bm25_index", index)
    await client.admin.command("ping")
    assert await cache.ping()
    await mongodb.db.document_versions.create_index("reindex_lock_key", unique=True, sparse=True)
    try:
        yield index
    finally:
        await client.drop_database(database)
        client.close()
        await cache.aclose()


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    class _StubLLM:
        provider_name = "test"

        def __init__(self, response: LLMResponse) -> None:
            self.chat = AsyncMock(return_value=response)

    stub = _StubLLM(LLMResponse(content="", model="test", provider="test"))
    monkeypatch.setattr(LLMFactory, "create", staticmethod(lambda provider=None: stub))
    monkeypatch.setattr(LLMFactory, "create_resilient", staticmethod(lambda provider=None: stub))
    return stub


async def test_concurrent_regenerate_calls_do_not_silently_lose_a_field_update(isolated_services):
    """BUG-015, fixed: `LegalDraftEngine.regenerate` now uses an optimistic
    -concurrency compare-and-swap (`DraftRepository.compare_and_swap`, keyed
    on a `version` counter) with one re-read-merge-retry cycle on conflict,
    instead of `DraftRepository.update_by_id`'s old unconditional
    `update_one($set)`. Two overlapping writers changing DIFFERENT fields of
    the same draft must both still land: whichever loses the initial race
    re-reads the winner's fresh document, re-merges its OWN requested field
    onto that fresh base, and retries -- so neither field is silently
    dropped, and neither call is ever told "Updated..." for a change that
    didn't survive. This was previously `xfail(strict=True)`, reproducing the
    bug deterministically; it now asserts the fix instead."""
    engine = LegalDraftEngine()
    created = await engine.generate(DraftGenerateRequest(draft_id="legal_notice", fields=_FIELDS))
    draft_id = created.draft_id
    assert draft_id

    # Two genuinely overlapping edits to two DIFFERENT fields of the SAME
    # draft, fired together via asyncio.gather so their reads race for real
    # over the real Mongo connection -- not two sequential calls that merely
    # look concurrent.
    result_a, result_b = await asyncio.gather(
        engine.regenerate(draft_id, {"applicant_mobile": "9000000001"}),
        engine.regenerate(draft_id, {"respondent_name": "Renamed Respondent Ltd"}),
    )
    assert result_a.status == "complete"
    assert result_b.status == "complete"

    persisted = await mongodb.db[LEGAL_DRAFTS].find_one({"_id": draft_id})
    assert persisted is not None
    # Both concurrent field changes must survive in the final persisted
    # document -- today only the last writer's change does, silently
    # discarding the other despite BOTH calls reporting "complete".
    assert persisted["fields"]["applicant_mobile"] == "9000000001"
    assert persisted["fields"]["respondent_name"] == "Renamed Respondent Ltd"
