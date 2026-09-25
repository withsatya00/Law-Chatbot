import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.exceptions import BadRequestError
from app.models.collections import UPLOADED_DOCUMENTS
from app.schemas.law_monitoring import MonitorRequest, MonitorReviewRequest
from app.services.law_monitoring import CHANGES, MONITORS, LawMonitoringService


def test_scoped_fingerprint_ignores_counter_but_detects_legal_table_changes():
    from app.services.law_monitoring import snapshot_fingerprint
    first = b'<table><tr><td>Rule A</td></tr></table><footer>1 visitor</footer>'
    counter = first.replace(b'1 visitor', b'2 visitors')
    changed = first.replace(b'Rule A', b'Rule B')
    assert snapshot_fingerprint(first, 'text/html', 'table') == snapshot_fingerprint(counter, 'text/html', 'table')
    assert snapshot_fingerprint(first, 'text/html', 'table') != snapshot_fingerprint(changed, 'text/html', 'table')
    with pytest.raises(ValueError):
        snapshot_fingerprint(b'<html>Maintenance</html>', 'text/html', 'table')


def matches(doc, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(doc, branch) for branch in expected):
                return False
            continue
        actual = doc.get(key)
        if isinstance(expected, dict):
            for op, value in expected.items():
                if op == "$exists" and (key in doc) != value:
                    return False
                if op == "$lte" and (actual is None or actual > value):
                    return False
                if op == "$in" and actual not in value:
                    return False
        elif actual != expected:
            return False
    return True


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, key, order):
        self.rows.sort(key=lambda row: row.get(key), reverse=order < 0)
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield deepcopy(row)
        return iterate()


class Collection:
    def __init__(self):
        self.rows = []

    async def create_index(self, *args, **kwargs):
        return None

    async def find_one(self, query, projection=None):
        return next((deepcopy(row) for row in self.rows if matches(row, query)), None)

    def find(self, query, projection=None):
        rows = [deepcopy(row) for row in self.rows if matches(row, query)]
        if projection:
            rows = [{k: v for k, v in row.items() if projection.get(k) != 0} for row in rows]
        return Cursor(rows)

    async def update_one(self, query, update, upsert=False):
        row = next((r for r in self.rows if matches(r, query)), None)
        if row is None:
            if not upsert:
                return SimpleNamespace(modified_count=0)
            row = {**query, **deepcopy(update.get("$setOnInsert", {}))}
            self.rows.append(row)
        row.update(deepcopy(update.get("$set", {})))
        for key in update.get("$unset", {}):
            row.pop(key, None)
        return SimpleNamespace(modified_count=1)

    async def find_one_and_update(self, query, update, **kwargs):
        found = await self.find_one(query)
        if found is None:
            return None
        await self.update_one({"_id": found["_id"]}, update)
        return await self.find_one({"_id": found["_id"]})

    async def count_documents(self, query):
        return sum(matches(row, query) for row in self.rows)


class Database(dict):
    def __missing__(self, key):
        self[key] = Collection()
        return self[key]


def service():
    db = Database()
    publisher = SimpleNamespace(update_jurisdiction_metadata=AsyncMock(return_value={"review_status": "approved", "chunks_updated": 2}))
    return LawMonitoringService(db, AsyncMock(return_value=(b"official text A", "text/html")), publisher, AsyncMock())


async def register(svc):
    return await svc.register(MonitorRequest(title="MP rules", url="https://example.gov.in/rules", state_code="MP", topic="property"), "admin")


def publication(**overrides):
    raw = {
        "decision": "publish", "notes": "Compared sections and affected versions.",
        "evidence_url": "https://example.gov.in/instrument.pdf", "affected_versions_reviewed": True,
        "publications": [{"document_id": "doc", "jurisdiction_metadata": {
            "issuing_level": "state", "applicability": "specific_states", "applicable_state_codes": ["MP"],
            "source_url": "https://example.gov.in/instrument.pdf", "effective_from": "2026-01-01",
            "verified_by": "forged-client-identity",
        }}],
    }
    return MonitorReviewRequest(**{**raw, **overrides})


@pytest.mark.parametrize("url", [
    "http://example.gov.in", "https://example.com", "https://example.gov.in.evil.com",
    "https://user:pass@example.gov.in", "https://example.gov.in:8443", "https://127.0.0.1",
])
def test_registration_rejects_nonofficial_targets(url):
    with pytest.raises(ValueError):
        MonitorRequest(title="x", url=url, topic="x")


def test_baseline_changes_and_returning_content_are_durable_and_idempotent():
    async def run():
        svc = service()
        monitor = await register(svc)
        assert (await svc.check(monitor["_id"]))["status"] == "changed"
        assert (await svc.check(monitor["_id"]))["status"] == "unchanged"
        for content in (b"B", b"official text A", b"B"):
            svc.fetcher.return_value = (content, "text/html")
            assert (await svc.check(monitor["_id"]))["status"] == "changed"
        assert len(svc.db[CHANGES].rows) == 4
        assert all(row["status"] == "pending_review" for row in svc.db[CHANGES].rows)
        assert "snapshot" not in (await svc.list_changes())[0]
        svc.publisher.update_jurisdiction_metadata.assert_not_awaited()
    asyncio.run(run())


def test_failure_preserves_baseline_and_retries_without_claiming_freshness():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        baseline = deepcopy(svc.db[MONITORS].rows[0])
        svc.fetcher.side_effect = TimeoutError()
        assert (await svc.check(monitor["_id"]))["status"] == "failed"
        result = svc.db[MONITORS].rows[0]
        assert result["checksum"] == baseline["checksum"]
        assert result["last_success_at"] == baseline["last_success_at"]
        assert result["next_check_at"] > result["last_checked_at"]
        assert (await svc.coverage())["monitors"][0]["monitoring_status"] == "stale"
    asyncio.run(run())


def test_active_lease_and_disabled_monitor_do_not_fetch():
    async def run():
        svc = service()
        monitor = await register(svc)
        row = svc.db[MONITORS].rows[0]
        row["lease_until"] = datetime.now(UTC) + timedelta(minutes=2)
        assert (await svc.check(monitor["_id"]))["status"] == "busy"
        row["enabled"] = False
        assert (await svc.check_due())["checked"] == 0
        svc.fetcher.assert_not_awaited()
    asyncio.run(run())


def test_checking_a_nonexistent_monitor_is_not_found_not_busy():
    """Security/correctness finding G4: the lease-acquisition query used to
    fail to match for two different reasons -- a real monitor that's
    disabled or already leased (genuinely "busy") and a `monitor_id` that
    was never registered at all. Conflating them reported 200 `{"status":
    "busy"}` for an id that will never succeed, however many times it's
    retried."""
    async def run():
        from app.core.exceptions import NotFoundError

        svc = service()
        with pytest.raises(NotFoundError):
            await svc.check("this-monitor-id-was-never-registered")
    asyncio.run(run())


def test_publish_stamps_authenticated_actor_and_preserves_rollback_metadata():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        svc.db[UPLOADED_DOCUMENTS].rows.append({"_id": "doc", "metadata": {"version_label": "original"}})
        change_id = svc.db[CHANGES].rows[0]["_id"]
        assert (await svc.review(change_id, publication(), "real-admin"))["status"] == "published"
        calls = svc.publisher.update_jurisdiction_metadata.await_args_list
        assert calls[0].args[1]["verification_status"] == "unverified"
        assert calls[1].args[1]["verified_by"] == "real-admin"
        assert svc.invalidator.await_count == 3
        assert svc.db[CHANGES].rows[0]["previous_metadata"]["doc"]["version_label"] == "original"
        with pytest.raises(BadRequestError):
            await svc.review(change_id, publication(), "real-admin")
    asyncio.run(run())


@pytest.mark.parametrize("private", [True, False])
def test_publication_rejects_private_or_incomplete_metadata_before_writing(private):
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        svc.db[UPLOADED_DOCUMENTS].rows.append({"_id": "doc", "metadata": {"owner_user_id": "user"} if private else {}})
        request = publication()
        if not private:
            request.publications[0].jurisdiction_metadata.pop("effective_from")
        with pytest.raises(BadRequestError):
            await svc.review(svc.db[CHANGES].rows[0]["_id"], request, "admin")
        svc.publisher.update_jurisdiction_metadata.assert_not_awaited()
        assert svc.db[CHANGES].rows[0]["status"] == "pending_review"
    asyncio.run(run())


def test_partial_publication_failure_requarantines_and_is_retryable():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        svc.db[UPLOADED_DOCUMENTS].rows.append({"_id": "doc"})
        svc.publisher.update_jurisdiction_metadata.side_effect = [{}, RuntimeError("index failure"), {}]
        change_id = svc.db[CHANGES].rows[0]["_id"]
        with pytest.raises(RuntimeError):
            await svc.review(change_id, publication(), "admin")
        assert svc.db[CHANGES].rows[0]["status"] == "publish_failed"
        assert svc.publisher.update_jurisdiction_metadata.await_args_list[-1].args[1]["verification_status"] == "unverified"
        svc.publisher.update_jurisdiction_metadata.side_effect = None
        assert (await svc.review(change_id, publication(), "admin"))["status"] == "published"
    asyncio.run(run())


def test_dismissal_never_touches_kb_or_legal_verification():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        request = publication(decision="dismiss", publications=[], affected_versions_reviewed=False)
        assert (await svc.review(svc.db[CHANGES].rows[0]["_id"], request, "admin"))["status"] == "dismissed"
        svc.publisher.update_jurisdiction_metadata.assert_not_awaited()
        assert "last_verified_at" not in svc.db[MONITORS].rows[0]
    asyncio.run(run())


@pytest.mark.parametrize("status,content_type,body", [
    (302, "text/html", b"redirect"), (503, "text/html", b"unavailable"),
    (200, "application/octet-stream", b"binary"), (200, "text/html", b""),
    (200, "text/html", b"too many bytes"),
])
def test_fetch_failures_do_not_become_baselines(monkeypatch, status, content_type, body):
    from app.services import law_monitoring as module

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", AsyncMock(return_value=[(0, 0, 0, "", ("8.8.8.8", 443))]))
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(status, headers={"content-type": content_type}, content=body))
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
        monkeypatch.setattr(module, "MAX_BYTES", 4)
        svc = service()
        svc.fetcher = module.fetch_official_snapshot
        monitor = await register(svc)
        assert (await svc.check(monitor["_id"]))["status"] == "failed"
        assert not svc.db[CHANGES].rows
        assert "last_success_at" not in svc.db[MONITORS].rows[0]
    asyncio.run(run())


def test_fetch_rejects_private_dns_and_keeps_verified_bytes(monkeypatch):
    from app.services import law_monitoring as module

    async def run():
        loop = asyncio.get_running_loop()
        dns = AsyncMock(return_value=[(0, 0, 0, "", ("127.0.0.1", 443))])
        monkeypatch.setattr(loop, "getaddrinfo", dns)
        with pytest.raises(ValueError, match="public"):
            await module.fetch_official_snapshot("https://example.gov.in/rules")
        dns.return_value = [(0, 0, 0, "", ("8.8.8.8", 443))]
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-original bytes"))
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
        assert await module.fetch_official_snapshot("https://example.gov.in/rules") == (b"%PDF-original bytes", "application/pdf")
    asyncio.run(run())


def test_register_is_idempotent_and_coverage_names_unconfigured_states():
    async def run():
        svc = service()
        await register(svc)
        await register(svc)
        assert len(svc.db[MONITORS].rows) == 1
        coverage = await svc.coverage()
        assert "UP" in coverage["unconfigured_state_codes"]
        assert "MP" not in coverage["unconfigured_state_codes"]
        assert coverage["monitors"][0]["monitoring_status"] == "never_checked"
        assert (await svc.check_due())["changed"] == 1
        assert (await svc.check_due())["checked"] == 0
        assert (await svc.coverage())["monitors"][0]["monitoring_status"] == "review_pending"
    asyncio.run(run())


def test_publish_requires_explicit_affected_version_review():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        with pytest.raises(BadRequestError, match="affected versions"):
            await svc.review(svc.db[CHANGES].rows[0]["_id"], publication(affected_versions_reviewed=False), "admin")
        svc.publisher.update_jurisdiction_metadata.assert_not_awaited()
    asyncio.run(run())


def test_review_api_keeps_admin_authorization():
    from app.api.admin_phase3 import router
    from app.api.deps import require_admin
    assert any(dependency.dependency is require_admin for dependency in router.dependencies)
    routes = {route.path for route in router.routes}
    assert "/admin/phase3/law-updates/{change_id}/review" in routes
    assert "/admin/phase3/law-updates/{change_id}/snapshot" in routes


def test_cache_failure_refuses_to_start_publication():
    async def run():
        svc = service()
        monitor = await register(svc)
        await svc.check(monitor["_id"])
        svc.db[UPLOADED_DOCUMENTS].rows.append({"_id": "doc"})
        svc.invalidator.side_effect = ConnectionError("Redis unavailable")
        with pytest.raises(ConnectionError):
            await svc.review(svc.db[CHANGES].rows[0]["_id"], publication(), "admin")
        svc.publisher.update_jurisdiction_metadata.assert_not_awaited()
        assert svc.db[CHANGES].rows[0]["status"] == "pending_review"
    asyncio.run(run())


def test_keyword_metadata_refreshes_on_generation_change_in_each_worker(monkeypatch):
    from app.cache.redis_client import redis_client
    from app.rag.bm25_index import BM25Index

    async def run():
        redis = SimpleNamespace(get=AsyncMock(return_value=b"1"))
        monkeypatch.setattr(redis_client, "_client", redis)
        workers = [BM25Index(), BM25Index()]
        for index in workers:
            index._bm25 = object()  # a loaded, potentially stale disk snapshot
            monkeypatch.setattr(index, "_rebuild_from_mongo", AsyncMock())
            await index.ensure_current_generation()
            await index.ensure_current_generation()
            assert index._rebuild_from_mongo.await_count == 1
        redis.get.return_value = b"2"
        for index in workers:
            await index.ensure_current_generation()
            assert index._rebuild_from_mongo.await_count == 2
        redis.get.side_effect = ConnectionError("not fresh")
        with pytest.raises(ConnectionError):
            await workers[0].ensure_current_generation()
    asyncio.run(run())
