"""HTTP readiness, real middleware boundaries, queue retries and secret validation."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import router
from app.cache.redis_client import redis_client
from app.core.config import Settings, settings
from app.core.middleware import RateLimitMiddleware
from app.core.windows_runtime import _accessible_directory
from app.database.mongodb import mongodb
from app.services.kb_indexing_queue import KnowledgeBaseIndexingQueue


@pytest.mark.parametrize("db_ok,redis_ok,expected", [(True, True, 200), (False, True, 503),
                                                      (True, False, 503), (False, False, 503)])
def test_readiness_http_status(monkeypatch, db_ok, redis_ok, expected):
    monkeypatch.setattr(mongodb, "ping", AsyncMock(return_value=db_ok))
    monkeypatch.setattr(redis_client, "ping", AsyncMock(return_value=redis_ok))
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == expected
    assert response.json()["status"] == ("ready" if expected == 200 else "not_ready")


def test_rate_limit_returns_429_and_does_not_block_probes(monkeypatch):
    backend = SimpleNamespace(incr=AsyncMock(return_value=settings.rate_limit_per_minute + 1))
    monkeypatch.setattr(redis_client, "_client", backend)
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/chat")
    @app.get("/health/live")
    def endpoint():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/chat")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert client.get("/health/live").status_code == 200
    assert backend.incr.await_count == 1


def test_unreadable_optional_tool_directory_is_not_fatal(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("restricted tool installation")
    monkeypatch.setattr(Path, "is_dir", denied)
    assert not _accessible_directory(Path("optional-tool"))


@pytest.mark.parametrize("field,value", [
    ("jwt_secret_key", "REPLACE_WITH_A_STRONG_RANDOM_SECRET_MIN_32_CHARS"),
    ("secrets_encryption_key", "REPLACE_WITH_A_SECRET_MANAGER_VALUE_MIN_32_CHARS"),
    ("api_cors_origins", ["*"]),
])
def test_production_rejects_sample_secrets_and_wildcard_cors(field, value):
    config = Settings.model_construct(environment="production", jwt_secret_key="a" * 48,
                                     secrets_encryption_key="b" * 48)
    setattr(config, field, value)
    with pytest.raises(RuntimeError):
        config._validate_security()


def test_duplicate_jobs_coalesce_and_consumer_survives_failure():
    async def exercise():
        gate = asyncio.Event()
        calls = []

        async def process(staging_id, *_args):
            calls.append(staging_id)
            if staging_id == "first":
                await gate.wait()
                raise RuntimeError("injected service failure")

        queue = KnowledgeBaseIndexingQueue(service=SimpleNamespace(process=process))
        try:
            queue.enqueue("first", Path("first.pdf"), "first.pdf")
            await asyncio.sleep(0)
            queue.enqueue("first", Path("first.pdf"), "first.pdf")
            queue.enqueue("second", Path("second.pdf"), "second.pdf")
            gate.set()
            await asyncio.wait_for(queue.join(), 2)
            assert calls == ["first", "second"]
            await queue.close()
            queue.enqueue("third", Path("third.pdf"), "third.pdf")
            await asyncio.wait_for(queue.join(), 2)
            assert calls == ["first", "second", "third"]
        finally:
            await queue.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["oversize", "disconnect", "cancelled"])
def test_upload_failure_removes_partial_file(tmp_path, monkeypatch, failure):
    from app.core.exceptions import BadRequestError
    from app.utils.upload_storage import write_upload

    monkeypatch.setattr(settings, "max_upload_mb", 1)
    error = asyncio.CancelledError() if failure == "cancelled" else OSError("disconnected")
    parts = [b"a" * (1024 * 1024), b"b"] if failure == "oversize" else [b"partial", error]
    upload = SimpleNamespace(read=AsyncMock(side_effect=parts))
    destination = tmp_path / "upload.pdf"
    expected = BadRequestError if failure == "oversize" else type(error)
    with pytest.raises(expected):
        asyncio.run(write_upload(upload, destination))
    assert not destination.exists()


def test_upload_collision_preserves_existing_file(tmp_path):
    from app.utils.upload_storage import write_upload
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError):
        asyncio.run(write_upload(SimpleNamespace(read=AsyncMock()), destination))
    assert destination.read_bytes() == b"existing evidence"


def test_production_upload_requires_authenticated_owner(monkeypatch):
    from app.api.upload import router as upload_router
    from app.core.exceptions import install_exception_handlers
    monkeypatch.setattr(settings, "environment", "production")
    app = FastAPI()
    app.include_router(upload_router)
    install_exception_handlers(app)
    response = TestClient(app).post("/upload", files={"file": ("private.txt", b"private evidence")})
    assert response.status_code == 401
