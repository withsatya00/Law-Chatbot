"""Phase 4 security hardening: `/feedback` had no ownership check at all --
any caller could attach a rating/comment to any message by guessing its
`message_id`, and `rating` had no range validation beyond being an int.
Mirrors the draft/document ownership test pattern: monkeypatch the
repository classes used inside `app.api.feedback`, call the route function
directly.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.api import feedback as feedback_api
from app.core.exceptions import ForbiddenError, NotFoundError
from app.schemas.history import FeedbackRequest


def _patch_repos(monkeypatch: pytest.MonkeyPatch, entry: dict | None) -> tuple[AsyncMock, AsyncMock]:
    query_log = AsyncMock()
    query_log.find_by_id = AsyncMock(return_value=entry)
    query_log.attach_feedback = AsyncMock(return_value=True)
    monkeypatch.setattr(feedback_api, "QueryLogRepository", lambda: query_log)

    feedback_repo = AsyncMock()
    feedback_repo.insert = AsyncMock(return_value="feedback-id")
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda: feedback_repo)
    return query_log, feedback_repo


def test_rating_outside_one_to_five_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(session_id="s1", rating=0)
    with pytest.raises(ValidationError):
        FeedbackRequest(session_id="s1", rating=6)


def test_feedback_without_message_id_skips_ownership_check(monkeypatch: pytest.MonkeyPatch) -> None:
    query_log, feedback_repo = _patch_repos(monkeypatch, entry=None)
    request = FeedbackRequest(session_id="s1", rating=4, comment="good")
    result = asyncio.run(feedback_api.feedback(request))
    assert result["status"] == "recorded"
    query_log.attach_feedback.assert_not_awaited()
    feedback_repo.insert.assert_awaited_once()


def test_feedback_for_a_message_owned_by_this_session_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    query_log, feedback_repo = _patch_repos(monkeypatch, entry={"session_id": "s1"})
    request = FeedbackRequest(session_id="s1", message_id="msg-1", rating=5)
    result = asyncio.run(feedback_api.feedback(request))
    assert result["status"] == "recorded"
    query_log.attach_feedback.assert_awaited_once_with("msg-1", 5, None)
    feedback_repo.insert.assert_awaited_once()


def test_feedback_for_a_message_owned_by_a_different_session_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    query_log, feedback_repo = _patch_repos(monkeypatch, entry={"session_id": "victim-session"})
    request = FeedbackRequest(session_id="attacker-session", message_id="msg-1", rating=1)
    with pytest.raises(ForbiddenError):
        asyncio.run(feedback_api.feedback(request))
    query_log.attach_feedback.assert_not_awaited()
    feedback_repo.insert.assert_not_awaited()


def test_feedback_for_a_nonexistent_message_id_returns_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _query_log, feedback_repo = _patch_repos(monkeypatch, entry=None)
    request = FeedbackRequest(session_id="s1", message_id="does-not-exist", rating=3)
    with pytest.raises(NotFoundError):
        asyncio.run(feedback_api.feedback(request))
    feedback_repo.insert.assert_not_awaited()
