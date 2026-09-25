"""Phase 4 security hardening: `app/api/drafting.py` had zero ownership
enforcement on any route (edit/export/approve/lock/unlock/rollback/
versions/translate/history) -- any caller could act on any draft by id.
Mirrors the existing `DocumentService._ensure_document_access` pattern
(see `test_document_service_ownership.py`) and the same three-way
visibility rule: no owner stamped -> open; `user_id` stamped -> only that
exact authenticated user; `session_id`-only stamped -> only that exact
session.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api import drafting
from app.core.exceptions import ForbiddenError, NotFoundError
from app.schemas.drafting import (
    DraftEditRequest,
    DraftExportRequest,
    DraftHistoryRequest,
    DraftLifecycleRequest,
    DraftRollbackRequest,
)


def _draft(**overrides: object) -> dict[str, object]:
    base = {
        "_id": "draft-1",
        "draft_type": "police_complaint",
        "template_name": "Police Complaint",
        "language": "english",
        "session_id": "owner-session",
        "user_id": "user-A",
        "fields": {},
        "sections": {"Recipient": "content"},
        "lifecycle_state": "preview_ready",
    }
    base.update(overrides)
    return base


def _fake_engine(draft: dict[str, object] | None) -> MagicMock:
    engine = MagicMock()
    engine.drafts.find_by_id = AsyncMock(return_value=draft)
    engine.approve = AsyncMock(return_value="approved")
    engine.lock = AsyncMock(return_value="locked")
    engine.unlock = AsyncMock(return_value="preview_ready")
    engine.rollback = AsyncMock(return_value="rolled-back-response")
    engine.regenerate = AsyncMock(return_value="edited-response")
    engine.export = AsyncMock(return_value="C:/tmp/draft-1.pdf")
    engine.translate = AsyncMock(return_value="translated-response")
    engine.history = AsyncMock(return_value=[])
    engine.list_versions = AsyncMock(return_value=[])
    return engine


# ---------------------------------------------------------------------------
# `_ensure_draft_access` -- pure function, same shape as document ownership
# ---------------------------------------------------------------------------


def test_legacy_draft_with_no_owner_is_open() -> None:
    drafting._ensure_draft_access({}, authenticated_user_id=None, session_id=None)
    drafting._ensure_draft_access({}, authenticated_user_id="user-A", session_id="s1")


def test_user_owned_draft_allowed_for_its_owner_from_any_session() -> None:
    draft = _draft()
    drafting._ensure_draft_access(draft, authenticated_user_id="user-A", session_id="brand-new-session")


def test_user_owned_draft_denied_for_a_different_authenticated_user() -> None:
    draft = _draft()
    with pytest.raises(ForbiddenError):
        drafting._ensure_draft_access(draft, authenticated_user_id="user-B", session_id="owner-session")


def test_user_owned_draft_denied_for_anonymous_caller_even_with_matching_session() -> None:
    draft = _draft()
    with pytest.raises(ForbiddenError):
        drafting._ensure_draft_access(draft, authenticated_user_id=None, session_id="owner-session")


def test_session_only_draft_allowed_for_matching_session() -> None:
    draft = _draft(user_id=None)
    drafting._ensure_draft_access(draft, authenticated_user_id=None, session_id="owner-session")


def test_session_only_draft_denied_for_a_different_session() -> None:
    draft = _draft(user_id=None)
    with pytest.raises(ForbiddenError):
        drafting._ensure_draft_access(draft, authenticated_user_id=None, session_id="someone-elses-session")


# ---------------------------------------------------------------------------
# Route-level enforcement -- each mutating route must check access before
# acting, not just fetch-and-proceed.
# ---------------------------------------------------------------------------


def test_edit_draft_rejects_a_different_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(_draft())
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftEditRequest(draft_id="draft-1", target_field="applicant_name", new_value="Someone Else")
    with pytest.raises(ForbiddenError):
        asyncio.run(drafting.edit_draft(request, user_id="user-B"))
    engine.regenerate.assert_not_awaited()


def test_edit_draft_allows_the_owning_user(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(_draft())
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftEditRequest(draft_id="draft-1", target_field="applicant_name", new_value="Ramesh")
    asyncio.run(drafting.edit_draft(request, user_id="user-A"))
    engine.regenerate.assert_awaited_once()


def test_export_draft_rejects_a_non_owning_session(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(_draft(user_id=None, session_id="owner-session"))
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftExportRequest(draft_id="draft-1", session_id="attacker-session")
    with pytest.raises(ForbiddenError):
        asyncio.run(drafting.export_draft(request, user_id=None))
    engine.export.assert_not_awaited()


def test_successful_export_records_an_audit_trail_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed export is the security/compliance-relevant event -- someone
    downloaded a document dense with another person's PII -- but only the
    FAILURE path (`AuditService.event("export_failed", ...)`, a separate,
    best-effort telemetry collection) was ever recorded anywhere; a
    successful export left no trace of who exported what, when.
    """
    engine = _fake_engine(_draft())
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    calls: list[dict] = []

    async def fake_record(self: object, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(drafting.AuditService, "record", fake_record)
    request = DraftExportRequest(draft_id="draft-1", session_id="owner-session")

    asyncio.run(drafting.export_draft(request, user_id="user-A"))

    assert len(calls) == 1
    assert calls[0]["action"] == "draft_exported"
    assert calls[0]["resource_id"] == "draft-1"
    assert calls[0]["outcome"] == "success"
    assert calls[0]["actor_user_id"] == "user-A"


def test_approve_draft_missing_draft_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(None)
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftLifecycleRequest(draft_id="missing-draft")
    with pytest.raises(NotFoundError):
        asyncio.run(drafting.approve_draft(request, user_id="user-A"))


def test_lock_and_rollback_reject_cross_user_access(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(_draft())
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    with pytest.raises(ForbiddenError):
        asyncio.run(drafting.lock_draft(DraftLifecycleRequest(draft_id="draft-1"), user_id="user-B"))
    with pytest.raises(ForbiddenError):
        asyncio.run(
            drafting.rollback_draft(
                DraftRollbackRequest(draft_id="draft-1", version_number=1), user_id="user-B"
            )
        )
    engine.lock.assert_not_awaited()
    engine.rollback.assert_not_awaited()


def test_draft_history_ignores_client_supplied_user_id_when_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(None)
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftHistoryRequest(user_id="victim-user-id")
    asyncio.run(drafting.draft_history(request, user_id=None))
    forwarded_request = engine.history.await_args.args[0]
    assert forwarded_request.user_id is None


def test_draft_history_uses_authenticated_identity_not_client_supplied_one(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_engine(None)
    monkeypatch.setattr(drafting, "LegalDraftEngine", lambda: engine)
    request = DraftHistoryRequest(user_id="spoofed-other-user")
    asyncio.run(drafting.draft_history(request, user_id="real-authenticated-user"))
    forwarded_request = engine.history.await_args.args[0]
    assert forwarded_request.user_id == "real-authenticated-user"
