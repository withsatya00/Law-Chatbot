"""Regression tests for security finding N3 (GET /notarization/eligibility
had no authentication or ownership check at all).

Root cause: `notarization_eligibility` took only `draft_id` as a query
parameter and fetched the draft by id with no identity check whatsoever --
any caller, including an unauthenticated one, could probe an arbitrary
`draft_id` for its category/eligibility. Fixed to mirror `POST
/notarization/prepare`'s own established gate exactly: authentication
required, and a draft owned by a different account reports 404 (not 403),
so this can't be used to distinguish "not yours" from "doesn't exist"
either.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.api.notarization import notarization_eligibility
from app.core.exceptions import ForbiddenError, NotFoundError


def _draft(**overrides: object) -> dict[str, object]:
    base = {"_id": "draft-1", "draft_type": "rent_agreement", "user_id": "owner-user"}
    base.update(overrides)
    return base


def test_unauthenticated_caller_is_refused() -> None:
    with pytest.raises(ForbiddenError):
        asyncio.run(notarization_eligibility(draft_id="draft-1", user_id=None))


def test_a_different_authenticated_user_gets_not_found_not_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """404, not 403 -- matching `POST /notarization/prepare`'s own choice not
    to let this probe confirm a draft_id exists for someone else's account."""
    from app.drafting.engine import LegalDraftEngine

    monkeypatch.setattr(LegalDraftEngine, "__init__", lambda self: setattr(self, "drafts", AsyncMock()))
    engine_instance = LegalDraftEngine()
    engine_instance.drafts.find_by_id = AsyncMock(return_value=_draft(user_id="owner-user"))
    monkeypatch.setattr("app.api.notarization.LegalDraftEngine", lambda: engine_instance)

    with pytest.raises(NotFoundError):
        asyncio.run(notarization_eligibility(draft_id="draft-1", user_id="a-different-user"))


def test_the_owner_can_check_their_own_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.drafting.engine import LegalDraftEngine

    monkeypatch.setattr(LegalDraftEngine, "__init__", lambda self: setattr(self, "drafts", AsyncMock()))
    engine_instance = LegalDraftEngine()
    engine_instance.drafts.find_by_id = AsyncMock(return_value=_draft(user_id="owner-user"))
    monkeypatch.setattr("app.api.notarization.LegalDraftEngine", lambda: engine_instance)

    result = asyncio.run(notarization_eligibility(draft_id="draft-1", user_id="owner-user"))

    assert result["draft_id"] == "draft-1"
    assert "eligible" in result


def test_a_draft_with_no_recorded_owner_is_reachable_by_any_authenticated_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matches `POST /notarization/prepare`'s own `draft.get("user_id") and
    ...` check -- a draft created before ownership stamping existed, or via
    an anonymous session, must not become permanently unreachable."""
    from app.drafting.engine import LegalDraftEngine

    monkeypatch.setattr(LegalDraftEngine, "__init__", lambda self: setattr(self, "drafts", AsyncMock()))
    engine_instance = LegalDraftEngine()
    engine_instance.drafts.find_by_id = AsyncMock(return_value=_draft(user_id=None))
    monkeypatch.setattr("app.api.notarization.LegalDraftEngine", lambda: engine_instance)

    result = asyncio.run(notarization_eligibility(draft_id="draft-1", user_id="any-authenticated-user"))

    assert result["draft_id"] == "draft-1"
