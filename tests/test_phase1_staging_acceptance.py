"""Chained staging acceptance test: login -> chat -> follow-up -> upload ->
draft -> export, in one continuous session against real infrastructure.

Same opt-in convention and disposable-service pattern as
tests/test_phase1_live_services.py (PHASE1_LIVE_TESTS=1,
docker-compose.phase1-test.yml's Mongo :37017 / Redis :36379): real Mongo,
real Redis, real HTTP routing through the actual FastAPI routers. Only the
LLM is stubbed (via `LLMFactory.create`/`create_resilient`, the single choke
point every service -- ChatService, LegalDraftEngine, DraftFieldExtractor --
constructs its client through), so this proves the WIRING end to end
(auth, session/memory persistence, file storage, draft persistence, export
file generation, and that every artifact stays linked to the same
owner/session) without needing a real LLM budget or staging credentials.

This does not exercise real grounded-answer quality -- that is
tests/test_legal_benchmark.py's job. The isolated test database here starts
empty, so the chat turns below are expected to take the "no verified
context" decline path, which is itself a real and correct code path to prove
works under real infra.
"""
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.drafting import router as draft_router
from app.api.upload import router as upload_router
from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import install_exception_handlers
from app.database.mongodb import mongodb
from app.drafting.templates.base import structure_sections_for
from app.llm.base import LLMResponse
from app.llm.factory import LLMFactory
from app.models.collections import LEGAL_DRAFTS, UPLOADED_DOCUMENTS
from app.rag.bm25_index import BM25Index

pytestmark = pytest.mark.skipif(os.environ.get("PHASE1_LIVE_TESTS") != "1", reason="requires isolated Phase 1 services")


@pytest.fixture
async def isolated_services(monkeypatch, tmp_path):
    """Identical to test_phase1_live_services.py's fixture of the same name --
    duplicated rather than imported to keep this file self-contained, matching
    this repo's existing convention of one integration test file per fixture
    setup (there is no shared tests/conftest.py)."""
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


class _StubLLM:
    """A minimal, explicit stand-in for LLMProvider -- deliberately NOT an
    AsyncMock() instance, whose auto-generated child attributes (e.g. a bare
    `.model`) previously leaked into a Pydantic response field expecting a
    real string and broke JSON serialization."""

    provider_name = "test"

    def __init__(self, response: LLMResponse) -> None:
        self.chat = AsyncMock(return_value=response)


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """One choke point: every service builds its LLM client through
    LLMFactory.create()/create_resilient() (chat_service.py, drafting/engine.py,
    drafting/field_extraction.py). Patching both staticmethods here means every
    ChatService()/LegalDraftEngine()/DraftFieldExtractor() constructed inside
    the route handlers below -- fresh per request, not something this test can
    reach into directly -- gets the stub automatically."""
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    stub = _StubLLM(LLMResponse(content=sectioned_response, model="test", provider="test"))
    monkeypatch.setattr(LLMFactory, "create", staticmethod(lambda provider=None: stub))
    monkeypatch.setattr(LLMFactory, "create_resilient", staticmethod(lambda provider=None: stub))
    return stub


def _app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(draft_router)
    return app


async def test_login_chat_followup_upload_draft_export_chain(isolated_services, tmp_path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(_app()), base_url="http://test") as client:
        # 1. Register + login.
        email = f"staging-{uuid4().hex}@example.com"
        password = "Synthetic-Staging-Password-77"
        registered = await client.post(
            "/register", json={"email": email, "password": password, "full_name": "Staging Acceptance"}
        )
        assert registered.status_code == 200, registered.text
        user_id = registered.json()["user_id"]
        login = await client.post("/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}

        # 2. First chat turn opens a session.
        first = await client.post(
            "/chat", json={"question": "What are my rights if I am arrested?"}, headers=headers
        )
        assert first.status_code == 200, first.text
        session_id = first.json()["session_id"]
        assert session_id

        # 3. Follow-up turn reuses the same session -- proves memory continuity.
        second = await client.post(
            "/chat", json={"question": "What should I do next?", "session_id": session_id}, headers=headers
        )
        assert second.status_code == 200, second.text
        assert second.json()["session_id"] == session_id
        history_count = await mongodb.db.chats.count_documents({"session_id": session_id})
        assert history_count == 2, "both chat turns must be persisted against the same session"

        # 4. Upload a document into the same session.
        upload = await client.post(
            "/upload",
            data={"session_id": session_id},
            files={"file": ("evidence.txt", b"Sample uploaded evidence for the staging acceptance chain.")},
            headers=headers,
        )
        assert upload.status_code == 200, upload.text
        document_id = upload.json()["document_id"]
        uploaded_doc = await mongodb.db[UPLOADED_DOCUMENTS].find_one({"_id": document_id})
        assert uploaded_doc is not None
        assert uploaded_doc.get("owner_user_id") == user_id
        assert uploaded_doc.get("owner_session_id") == session_id

        # 5. Draft a document in the same session (deterministic field
        # extraction, no LLM needed for extraction -- see
        # test_draft_conversation.py's identical pattern).
        first_draft_turn = await client.post(
            "/draft",
            json={
                "session_id": session_id,
                "message": (
                    "I want to write a police complaint. My name is Ramesh Kumar, mobile 9876543210, "
                    "police station Hazratganj, incident happened at MG Road."
                ),
            },
            headers=headers,
        )
        assert first_draft_turn.status_code == 200, first_draft_turn.text
        draft_info = first_draft_turn.json()["draft"]
        assert draft_info["template_id"] == "police_complaint"

        second_draft_turn = await client.post(
            "/draft",
            json={
                "session_id": session_id,
                "message": (
                    "The facts are: goods were stolen from my shop on 1 January 2026. I want the police to "
                    "register an FIR and investigate. My address is 12 MG Road, Hazratganj."
                ),
            },
            headers=headers,
        )
        assert second_draft_turn.status_code == 200, second_draft_turn.text
        draft_info = second_draft_turn.json()["draft"]

        if draft_info["stage"] == "collecting":
            # Same fallback as test_draft_conversation.py's unit-level
            # equivalent: fill whatever the deterministic extractor still
            # missed directly in the real, persisted session memory, then
            # send one closing turn to reach "preview".
            from app.memory.store import ConversationMemoryStore

            memory_store = ConversationMemoryStore()
            memory = await memory_store.load(session_id)
            fields = dict(memory.get("draft_fields") or {})
            for field_key in draft_info["missing_fields"]:
                fields[field_key] = f"Test value for {field_key}"
            await memory_store.update(session_id, draft_fields=fields)
            third_draft_turn = await client.post(
                "/draft", json={"session_id": session_id, "message": "here you go"}, headers=headers
            )
            assert third_draft_turn.status_code == 200, third_draft_turn.text
            draft_info = third_draft_turn.json()["draft"]

        assert draft_info["stage"] == "preview", draft_info
        draft_id = draft_info["draft_id"]
        assert draft_id

        # 6. Export the completed draft.
        export = await client.post(
            "/draft/export",
            json={"draft_id": draft_id, "session_id": session_id, "format": "txt", "watermark": False},
            headers=headers,
        )
        assert export.status_code == 200, export.text
        assert export.content, "export must return a non-empty file body"

        draft_record = await mongodb.db[LEGAL_DRAFTS].find_one({"_id": draft_id})
        assert draft_record is not None
        assert draft_record.get("session_id") == session_id
        # BUG-014, fixed: `DraftConversationEngine._finalize_generation_reply`
        # (app/drafting/conversation.py, the single choke point that actually
        # persists version 1 of a draft) now reads `memory["owner_user_id"]`
        # -- already reliably set by this point via `_claim_session_if_
        # unowned` (chat_service.py) / the `/draft` route (app/api/
        # drafting.py) -- into `DraftGenerateRequest.user_id`. Previously this
        # was always `None` for a draft created through the conversational
        # flow, making `session_id` a de facto bearer credential across
        # accounts (live-reproduced, QA session 4: a second real, differently
        # -authenticated user could read this draft by supplying the SAME
        # session_id). `ensure_draft_access` (app/services/
        # draft_management.py:40-50) already correctly enforces `user_id`
        # ownership once it is actually present -- it just had nothing to
        # check before. This assertion is the acceptance test for the fix:
        # it must equal the authenticated creator's own user_id, never None,
        # for a draft created while logged in.
        assert draft_record.get("user_id") == user_id


async def test_second_authenticated_user_cannot_read_first_users_draft_via_session_id(isolated_services):
    """BUG-014's exact live repro (QA session 4), reproduced against real
    infrastructure: two genuinely different, authenticated accounts. Before
    the fix, `session_id` was a de facto bearer credential for a draft's
    entire lifetime regardless of authentication -- User B's own valid,
    different JWT did nothing to stop them from reading User A's draft once
    they had (or guessed) A's session_id, because the draft's `user_id` was
    never stamped by the one code path that actually creates the record.
    """
    async with httpx.AsyncClient(transport=httpx.ASGITransport(_app()), base_url="http://test") as client:
        # User A: register, login, create and complete a draft via the
        # conversational /draft flow (the ONLY engine /chat and /draft --
        # the two routes real users and the Streamlit frontend actually use
        # -- create drafts through).
        email_a = f"staging-a-{uuid4().hex}@example.com"
        password_a = "Synthetic-Staging-Password-A1"
        registered_a = await client.post(
            "/register", json={"email": email_a, "password": password_a, "full_name": "User A"}
        )
        assert registered_a.status_code == 200, registered_a.text
        login_a = await client.post("/login", json={"email": email_a, "password": password_a})
        assert login_a.status_code == 200, login_a.text
        headers_a = {"Authorization": "Bearer " + login_a.json()["access_token"]}
        session_a = "phase1-usera-" + uuid4().hex

        first_turn = await client.post(
            "/draft",
            json={
                "session_id": session_a,
                "message": (
                    "I want to write a police complaint. My name is Alice Owner, mobile 9876500000, "
                    "police station Hazratganj, incident at MG Road."
                ),
            },
            headers=headers_a,
        )
        assert first_turn.status_code == 200, first_turn.text
        draft_info = first_turn.json()["draft"]
        if draft_info["stage"] == "collecting":
            from app.memory.store import ConversationMemoryStore

            memory_store = ConversationMemoryStore()
            memory = await memory_store.load(session_a)
            fields = dict(memory.get("draft_fields") or {})
            for field_key in draft_info["missing_fields"]:
                fields[field_key] = f"Test value for {field_key}"
            await memory_store.update(session_a, draft_fields=fields)
            second_turn = await client.post(
                "/draft", json={"session_id": session_a, "message": "here you go"}, headers=headers_a
            )
            assert second_turn.status_code == 200, second_turn.text
            draft_info = second_turn.json()["draft"]
        assert draft_info["stage"] == "preview", draft_info
        draft_id = draft_info["draft_id"]
        assert draft_id

        # The fix under test: the draft record is actually owned by User A now.
        draft_record = await mongodb.db[LEGAL_DRAFTS].find_one({"_id": draft_id})
        assert draft_record is not None
        alice_user_id = registered_a.json()["user_id"]
        assert draft_record.get("user_id") == alice_user_id

        # User B: a completely different, genuinely authenticated account.
        email_b = f"staging-b-{uuid4().hex}@example.com"
        password_b = "Synthetic-Staging-Password-B1"
        registered_b = await client.post(
            "/register", json={"email": email_b, "password": password_b, "full_name": "User B"}
        )
        assert registered_b.status_code == 200, registered_b.text
        login_b = await client.post("/login", json={"email": email_b, "password": password_b})
        assert login_b.status_code == 200, login_b.text
        headers_b = {"Authorization": "Bearer " + login_b.json()["access_token"]}

        # User B's own valid token, no session_id at all: correctly refused
        # both before and after the fix (this was never the gap).
        no_session = await client.get(f"/draft/{draft_id}/versions", headers=headers_b)
        assert no_session.status_code == 403, no_session.text

        # User B's own valid token PLUS User A's session_id: this is the
        # exact live repro. Before the fix this returned 200 with the full
        # version history, because `ensure_draft_access` fell back to
        # session-only ownership since `user_id` was never stamped. Now that
        # the draft is genuinely owned by Alice's user_id, User B's own
        # (different) authenticated identity must still be refused,
        # regardless of knowing Alice's session_id.
        with_alice_session = await client.get(
            f"/draft/{draft_id}/versions", params={"session_id": session_a}, headers=headers_b
        )
        assert with_alice_session.status_code == 403, with_alice_session.text

        # Sanity check: Alice's OWN token, from a brand-new session she never
        # used before, still works -- ownership follows the account, not the
        # session, per ensure_draft_access's own contract.
        alice_new_session = await client.get(
            f"/draft/{draft_id}/versions",
            params={"session_id": "phase1-usera-different-" + uuid4().hex},
            headers=headers_a,
        )
        assert alice_new_session.status_code == 200, alice_new_session.text
