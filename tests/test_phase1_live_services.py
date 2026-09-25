"""Opt-in integration against ONLY docker-compose.phase1-test.yml's disposable ports.

PHASE1_LIVE_TESTS=1 enables these. Real MongoDB, Redis, auth, indexing and BM25;
embedding vectors are deterministic test doubles, with no paid model calls.
"""
import os
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from app.api.auth import router as auth_router
from app.api.deps import get_current_user_claims
from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import UnauthorizedError, install_exception_handlers
from app.database.mongodb import mongodb
from app.rag.bm25_index import BM25Index
from app.rag.pipeline import IndexingPipeline
from app.rag.vector_store import MongoVectorStore

pytestmark = pytest.mark.skipif(os.environ.get("PHASE1_LIVE_TESTS") != "1", reason="requires isolated Phase 1 services")


@pytest.fixture
async def isolated_services(monkeypatch, tmp_path):
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
        # The database name is generated here and the client has a fixed test-only port.
        await client.drop_database(database)
        client.close()
        await cache.aclose()


async def test_real_registration_login_and_revocation(isolated_services):
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(auth_router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        payload = {"email": f"phase1-{uuid4().hex}@example.com", "password": "Synthetic-Test-Password-92",
                   "full_name": "Phase One Test"}
        registered = await client.post("/register", json=payload)
        assert registered.status_code == 200, registered.text
        login = await client.post("/login", json={"email": payload["email"], "password": payload["password"]})
        assert login.status_code == 200
        header = "Bearer " + login.json()["access_token"]
        claims = await get_current_user_claims(header)
        assert claims["sub"] == registered.json()["user_id"]
        assert (await client.post("/logout", headers={"Authorization": header})).status_code == 200
        with pytest.raises(UnauthorizedError):
            await get_current_user_claims(header)


class TestEmbeddings:
    async def embed_batch(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.mark.parametrize("failure_point", ["bm25_switch", "version_record"])
async def test_real_reindex_rollback_then_success(isolated_services, tmp_path, monkeypatch, failure_point):
    from unittest.mock import AsyncMock

    source = tmp_path / "sample.txt"
    source.write_text("Section 1\nThe sample agreement requires written notice before termination. " * 8)
    pipeline = IndexingPipeline(embeddings=TestEmbeddings())
    old_id, _, _ = await pipeline.index_file(source)
    source.write_text("Section 1\nThe revised agreement requires written consent before termination. " * 8)
    with monkeypatch.context() as fault:
        if failure_point == "version_record":
            fault.setattr(pipeline.versions, "activate", AsyncMock(side_effect=RuntimeError("injected metadata outage")))
        else:
            original = isolated_services.mark_document_status
            failed = False

            async def mark(ids, status):
                nonlocal failed
                await original(ids, status)
                if status == "superseded" and not failed:
                    failed = True
                    raise RuntimeError("injected BM25 failure after mutation")
            fault.setattr(isolated_services, "mark_document_status", mark)
        with pytest.raises(RuntimeError):
            await pipeline.index_file(source)
    active = await mongodb.db.embeddings_metadata.distinct("document_id", {"metadata.document_status": "active"})
    assert active == [old_id]
    assert await mongodb.db.document_versions.count_documents({"reindex_lock_key": {"$exists": True}}) == 0
    old_results = await pipeline.vector_store.search([1.0, 0.0, 0.0], "notice", 6, {})
    assert old_results and all("written notice" in c.text for c in old_results)
    new_id, _, _ = await pipeline.index_file(source)
    active = await mongodb.db.embeddings_metadata.distinct("document_id", {"metadata.document_status": "active"})
    assert active == [new_id] and new_id != old_id
    assert await mongodb.db.embeddings_metadata.count_documents({"document_id": old_id}) == 0


async def test_real_same_filename_private_versions_are_isolated(isolated_services, tmp_path):
    source = tmp_path / "contract.txt"
    pipeline = IndexingPipeline(embeddings=TestEmbeddings())
    source.write_text("Section 1\nAlice confidential tenancy agreement requires written notice. " * 8)
    alice, _, _ = await pipeline.index_file(source, owner_user_id="alice")
    source.write_text("Section 1\nBob private purchase agreement requires written consent. " * 8)
    bob, _, _ = await pipeline.index_file(source, owner_user_id="bob")
    assert alice != bob
    assert await mongodb.db.embeddings_metadata.count_documents({"document_id": alice, "metadata.document_status": "active"}) > 0
    results = await MongoVectorStore().search([1.0, 0.0, 0.0], "agreement", 10,
                                            {"owner_user_id": [None, "bob"], "owner_session_id": [None]})
    assert results and all(c.metadata.get("owner_user_id") == "bob" for c in results)


async def test_interrupted_publication_recovers_and_can_retry(isolated_services, tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from app.services.reindex_recovery import recover_interrupted_reindexes

    source = tmp_path / "interrupted.txt"
    source.write_text("Section 1\nThe original agreement requires written notice before termination. " * 8)
    pipeline = IndexingPipeline(embeddings=TestEmbeddings())
    old_id, _, _ = await pipeline.index_file(source)
    source.write_text("Section 1\nThe updated agreement requires written consent before termination. " * 8)
    with monkeypatch.context() as fault:
        fault.setattr(pipeline.versions, "activate", AsyncMock(side_effect=asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await pipeline.index_file(source)
    preview = await recover_interrupted_reindexes()
    assert len(preview) == 1 and preview[0]["status"] == "pending"
    with pytest.raises(ValueError):
        await recover_interrupted_reindexes(apply=True)
    recovered = await recover_interrupted_reindexes(apply=True, writers_stopped=True)
    assert recovered[0]["status"] == "recovered_retry_source"
    active = await mongodb.db.embeddings_metadata.distinct("document_id", {"metadata.document_status": "active"})
    assert active == [old_id]
    assert await recover_interrupted_reindexes(apply=True, writers_stopped=True) == []
    new_id, _, _ = await pipeline.index_file(source)
    assert new_id != old_id


async def test_draft_and_upload_cannot_mutate_another_users_session(isolated_services):
    from app.api.drafting import router as draft_router
    from app.api.upload import router as upload_router
    from app.core.security import Role, create_access_token
    from app.memory.store import ConversationMemoryStore

    session = "phase1-" + uuid4().hex
    await ConversationMemoryStore().update(session, owner_user_id="alice")
    app = FastAPI()
    app.include_router(draft_router)
    app.include_router(upload_router)
    install_exception_handlers(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        for headers in ({}, {"Authorization": "Bearer " + create_access_token("bob", Role.user)}):
            draft = await client.post("/draft", json={"session_id": session, "message": "draft a notice"}, headers=headers)
            assert draft.status_code == 403, draft.text
            upload = await client.post("/upload", data={"session_id": session},
                                       files={"file": ("private.txt", b"private evidence")}, headers=headers)
            assert upload.status_code == 403, upload.text
