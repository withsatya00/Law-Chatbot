"""The admin-facing KB staging surface: the `status=all` listing, the
dashboard's separated ledger/disk counts, and the fact that every mutating
review action is admin-only.

Authorization is asserted against the real router (`app.api.admin.router`)
rather than a live server: `require_admin` is a router-level dependency, so
what matters is that every one of these routes is on that router and that the
dependency rejects a non-admin role.
"""

import asyncio
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.api import admin as admin_api
from app.api.deps import require_admin
from app.core.config import settings
from app.core.exceptions import ForbiddenError
from app.repositories import base as base_repository
from app.repositories.kb_staging import KnowledgeBaseStagingRepository
from app.services import admin_operations

_SCRATCH_ROOT = Path("storage") / "_test_kb_admin_review"

_RECORDS: list[dict[str, Any]] = [
    {"_id": "1", "original_filename": "a.pdf", "status": "pending", "current_path": "a.pdf"},
    {"_id": "2", "original_filename": "b.pdf", "status": "processing", "current_path": "b.pdf"},
    {"_id": "3", "original_filename": "c.pdf", "status": "indexed", "current_path": "c.pdf"},
    {"_id": "4", "original_filename": "d.pdf", "status": "duplicate", "current_path": "d.pdf"},
    {"_id": "5", "original_filename": "e.pdf", "status": "failed", "current_path": "e.pdf"},
    {
        "_id": "6", "original_filename": "f.pdf", "status": "needs_review", "current_path": "f.pdf",
        "resolution": "closed_missing_file",
    },
    {
        "_id": "7",
        "original_filename": "vanished.pdf",
        "status": "needs_review",
        "current_path": "vanished.pdf",
        "path_missing": True,
    },
]


class _FakeCursor:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def sort(self, *_args: Any) -> "_FakeCursor":
        return self

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for item in self._items:
                yield dict(item)

        return _gen()


class _FakeCollection:
    def find(self, query: dict[str, Any]) -> _FakeCursor:
        if not query:
            return _FakeCursor(_RECORDS)
        if "path_missing" in query:
            return _FakeCursor([r for r in _RECORDS if r.get("path_missing") and "resolution" not in r])
        wanted = query["status"]
        allowed = wanted["$in"] if isinstance(wanted, dict) else [wanted]
        records = [r for r in _RECORDS if r["status"] in allowed]
        if query.get("resolution") == {"$exists": False}:
            records = [r for r in records if "resolution" not in r]
        return _FakeCursor(records)

    async def count_documents(self, query: dict[str, Any]) -> int:
        if not query:
            return len(_RECORDS)
        return sum(1 for r in _RECORDS if r["status"] == query.get("status"))


class _FakeDb:
    def __getitem__(self, _name: str) -> _FakeCollection:
        return _FakeCollection()


class _FakeMongo:
    """`mongodb.db` is a property that raises when unconnected, so the whole
    accessor is swapped rather than the attribute behind it."""

    db = _FakeDb()


@contextmanager
def _fake_mongo():
    # Both call sites read the module-level `mongodb`: the service layer for
    # the listing query, and `MongoRepository.collection` for the counts.
    fake = _FakeMongo()
    originals = (admin_operations.mongodb, base_repository.mongodb)
    admin_operations.mongodb = fake  # type: ignore[assignment]
    base_repository.mongodb = fake  # type: ignore[assignment]
    try:
        yield
    finally:
        admin_operations.mongodb, base_repository.mongodb = originals  # type: ignore[assignment]


# ---- 12. status=all returns every status ---------------------------------------


def test_status_all_returns_every_status() -> None:
    with _fake_mongo():
        result = asyncio.run(admin_operations.staging_records("all"))

    assert result["status_filter"] == "all"
    assert result["count"] == len(_RECORDS)
    # A path_missing finding must stay visible rather than quietly vanishing.
    assert any(r["_id"] == "7" and r.get("path_missing") for r in result["records"])
    assert {r["status"] for r in result["records"]} == {
        "pending",
        "processing",
        "indexed",
        "duplicate",
        "failed",
        "needs_review",
    }


def test_default_listing_shows_only_actionable_records() -> None:
    with _fake_mongo():
        result = asyncio.run(admin_operations.staging_records())

    assert result["status_filter"] == "actionable"
    assert {r["status"] for r in result["records"]} == {"pending", "processing", "failed", "needs_review"}
    assert all(record["_id"] != "6" for record in result["records"])


def test_needs_review_filter_excludes_resolved_missing_file_findings() -> None:
    with _fake_mongo():
        result = asyncio.run(admin_operations.staging_records("needs_review"))

    assert [record["_id"] for record in result["records"]] == ["7"]


def test_every_listed_record_reports_its_real_location_and_existence() -> None:
    with _fake_mongo():
        result = asyncio.run(admin_operations.staging_records("all"))

    # These ledger paths do not exist on disk, and the listing says so rather
    # than presenting a stale path as if the file were still there.
    assert all(record["file_exists"] is False for record in result["records"])
    assert all(record["current_path"] is not None for record in result["records"])


# ---- 13. Dashboard: ledger and disk counted separately --------------------------


def test_dashboard_reports_ledger_and_disk_counts_separately() -> None:
    staging = _SCRATCH_ROOT / "staging"
    kb = _SCRATCH_ROOT / "kb"
    review_pending = _SCRATCH_ROOT / "review" / "pending"
    for directory in (staging, kb, review_pending):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (staging / f"s{index}.pdf").write_bytes(b"staged")
    (kb / "indexed.pdf").write_bytes(b"kb")
    (review_pending / "r.pdf").write_bytes(b"review")

    originals = (settings.kb_staging_dir, settings.knowledge_base_dir, settings.kb_review_dir, settings.archive_dir)
    settings.kb_staging_dir = staging
    settings.knowledge_base_dir = kb
    settings.kb_review_dir = _SCRATCH_ROOT / "review"
    settings.archive_dir = _SCRATCH_ROOT / "archive"
    try:
        with _fake_mongo():
            board = asyncio.run(admin_operations.kb_dashboard())
    finally:
        (
            settings.kb_staging_dir,
            settings.knowledge_base_dir,
            settings.kb_review_dir,
            settings.archive_dir,
        ) = originals
        shutil.rmtree(_SCRATCH_ROOT, ignore_errors=True)

    # Ledger: one row per status, all six statuses reported distinctly.
    assert board["ledger"] == {
        "pending": 1,
        "processing": 1,
        "indexed": 1,
        "duplicate": 1,
        "failed": 1,
        "needs_review": 2,
        "stale_or_missing": 2,
        # Ledger findings, counted apart from the physical staging files below.
        "path_missing": 1,
        "total": 7,
    }
    # Disk: what is actually there, which is a different number from the
    # ledger's -- conflating the two is what hid 68 untracked staging files.
    assert board["disk"]["staging_files"] == 3
    assert board["disk"]["knowledge_base_files"] == 1
    assert board["disk"]["review_pending_files"] == 1
    assert board["disk"]["review_queue_files"] == 1
    assert board["disk"]["staging_files"] != board["ledger"]["pending"]


def test_dashboard_counts_every_status_the_repository_knows() -> None:
    assert set(KnowledgeBaseStagingRepository.STATUSES) == {
        "pending",
        "processing",
        "indexed",
        "duplicate",
        "failed",
        "needs_review",
    }


# ---- 14. Only admins can approve / retry / archive / reconcile ------------------


def _route_paths() -> dict[str, set[str]]:
    paths: dict[str, set[str]] = {}
    for route in admin_api.router.routes:
        paths.setdefault(route.path, set()).update(route.methods)  # type: ignore[attr-defined]
    return paths


def test_every_mutating_kb_review_route_is_on_the_admin_guarded_router() -> None:
    paths = _route_paths()
    for path in (
        "/admin/knowledge-base/staging/{staging_id}/approve",
        "/admin/knowledge-base/staging/{staging_id}/retry",
        "/admin/knowledge-base/staging/{staging_id}/archive",
        "/admin/knowledge-base/reconcile-staging",
        "/admin/knowledge-base/reconcile-staging/apply-manifest",
        "/admin/knowledge-base/backfill-uploads",
        "/admin/knowledge-base/staging/{staging_id}/close-missing",
        "/admin/knowledge-base/staging/{staging_id}/resource",
    ):
        assert path in paths, path
        assert "POST" in paths[path]

    # The guard is declared once, on the router, so no handler can be added
    # without it.
    dependencies = [dependency.dependency for dependency in admin_api.router.dependencies]
    assert require_admin in dependencies


def test_a_non_admin_role_is_rejected_by_the_router_guard() -> None:
    with pytest.raises(ForbiddenError):
        asyncio.run(require_admin(claims={"sub": "u1", "role": "user"}))

    claims = asyncio.run(require_admin(claims={"sub": "admin-1", "role": "admin"}))
    assert claims["role"] == "admin"


def test_mutating_actions_are_audit_logged() -> None:
    """Each admin decision that changes the ledger records who did what."""
    recorded: list[dict[str, Any]] = []

    class _FakeAudit:
        async def insert(self, document: dict[str, Any]) -> str:
            recorded.append(document)
            return "audit-1"

    with patch.object(admin_api, "AuditLogRepository", _FakeAudit):
        asyncio.run(admin_api._audit("kb_approve_needs_review", "admin-1", staging_id="s1"))

    assert recorded[0]["action"] == "kb_approve_needs_review"
    assert recorded[0]["actor_user_id"] == "admin-1"
    assert recorded[0]["resource"] == "knowledge_base"
    assert recorded[0]["details"] == {"staging_id": "s1"}


def test_the_manifest_audit_endpoints_are_read_only() -> None:
    paths = _route_paths()
    for path in (
        "/admin/knowledge-base/reconciliation/manifests",
        "/admin/knowledge-base/reconciliation/manifests/{name}",
    ):
        assert path in paths, path
        # Auditing history must not be able to change it.
        assert paths[path] <= {"GET", "HEAD"}
