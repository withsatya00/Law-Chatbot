"""Chat-first authentication and the KB review flow.

Two gaps this covers, both introduced by the chat-first refactor:

* the Streamlit login form was deleted along with the admin pages, so the
  client had no way to obtain a token at all -- `POST /chat` went out
  unauthenticated, `_try_refresh_admin_token` rotated tokens no UI flow ever
  set, and `admin_token`/`access_token` were read in different places;
* the admin KB workflow could LIST files needing review but had no way to
  approve or archive one, so the review queue was read-only in chat.

`streamlit_app/auth_client.py` takes the session mapping as an argument
rather than importing `streamlit`, so everything here runs against a plain
dict and a stub transport. No test in this module opens a socket.
"""

import asyncio
import importlib.util
from pathlib import Path
from typing import Any, ClassVar, Self
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import app.chatops.orchestrator  # noqa: F401  - importing registers the workflows
from app.chatops.base import WorkflowContext
from app.chatops.registry import best_match
from app.chatops.workflows.admin import AdminKnowledgeBaseWorkflow
from app.core.security import Role
from app.schemas.auth import RegisterRequest


def _load_auth_client() -> Any:
    """Loaded by path rather than by putting `streamlit_app/` on `sys.path`:
    that directory contains `app.py`, which would shadow the `app` PACKAGE and
    break every `app.chatops...` import in this module."""
    path = Path(__file__).resolve().parents[1] / "streamlit_app" / "auth_client.py"
    spec = importlib.util.spec_from_file_location("streamlit_auth_client", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auth_client = _load_auth_client()

API = "http://testserver"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# A stub transport: records every request and replays queued responses.
# ---------------------------------------------------------------------------


class _StubClient:
    """Stands in for `httpx.Client`. `responses` is consumed in order."""

    def __init__(self, responses: list[httpx.Response], sent: list[dict[str, Any]], **_: Any) -> None:
        self._responses = responses
        self._sent = sent

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self._sent.append({"method": method, "url": url, **kwargs})
        return self._responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)


def _factory(*responses: httpx.Response) -> tuple[Any, list[dict[str, Any]]]:
    queued, sent = list(responses), []

    def make(**kwargs: Any) -> _StubClient:
        return _StubClient(queued, sent, **kwargs)

    return make, sent


def _json(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status_code=status, json=payload, request=httpx.Request("POST", API))


def _tokens(access: str = "access-1", refresh: str = "refresh-1", role: str = "user") -> dict[str, Any]:
    return {"access_token": access, "refresh_token": refresh, "role": role}


# ---------------------------------------------------------------------------
# 1. Login and logout
# ---------------------------------------------------------------------------


def test_login_stores_the_token_pair_under_one_naming_convention() -> None:
    state: dict[str, Any] = {}
    make, sent = _factory(_json(200, _tokens(role="admin")))

    ok, message = auth_client.login(state, API, "admin@example.com", "correct-horse", client_factory=make)

    assert ok is True
    assert state["access_token"] == "access-1"
    assert state["refresh_token"] == "refresh-1"
    assert state["user_role"] == "admin"
    # The defect this pins: two conventions were read in different places.
    assert "admin_token" not in state
    assert "admin_refresh_token" not in state
    assert sent[0]["url"] == f"{API}/login"
    assert message


def test_login_never_stores_or_returns_the_password() -> None:
    state: dict[str, Any] = {}
    make, _ = _factory(_json(200, _tokens()))
    _, message = auth_client.login(state, API, "u@example.com", "sup3r-secret-pw", client_factory=make)

    assert "sup3r-secret-pw" not in message
    assert "sup3r-secret-pw" not in repr(state)


def test_a_rejected_login_leaves_the_session_signed_out_and_says_nothing_specific() -> None:
    state: dict[str, Any] = {}
    make, _ = _factory(_json(401, {"detail": "no such user"}))
    ok, message = auth_client.login(state, API, "u@example.com", "wrong-password", client_factory=make)

    assert ok is False
    assert auth_client.is_authenticated(state) is False
    # Identical for "no such account" and "wrong password" -- a differentiated
    # message is an account-existence oracle.
    assert "no such user" not in message


def test_logout_revokes_the_refresh_token_and_clears_every_owned_key() -> None:
    state: dict[str, Any] = {
        "access_token": "a", "refresh_token": "r", "user_role": "admin", "user_email": "a@b.c",
    }
    make, sent = _factory(_json(200, {"status": "ok"}))

    auth_client.logout(state, API, client_factory=make)

    assert sent[0]["url"] == f"{API}/logout"
    assert sent[0]["json"] == {"refresh_token": "r"}
    assert state == {}


def test_logout_clears_the_session_even_when_the_server_is_unreachable() -> None:
    state: dict[str, Any] = {"access_token": "a", "refresh_token": "r", "user_role": "admin"}

    def make(**_: Any) -> Any:
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        make()
    # The real call path swallows the transport error; the session must still clear.
    class _Failing(_StubClient):
        def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("down")

    auth_client.logout(state, API, client_factory=lambda **_: _Failing([], []))
    assert state == {}


# ---------------------------------------------------------------------------
# 2. Authorization forwarding
# ---------------------------------------------------------------------------


def test_the_bearer_header_is_built_from_the_access_token() -> None:
    assert auth_client.auth_headers({"access_token": "abc"}) == {"Authorization": "Bearer abc"}


def test_a_signed_out_session_sends_no_authorization_header_at_all() -> None:
    """Not an empty `Bearer `: that turns an anonymous-but-allowed call into a 401."""
    assert auth_client.auth_headers({}) == {}


def test_chat_carries_the_bearer_token() -> None:
    state = {"access_token": "tok-1", "refresh_token": "r"}
    make, sent = _factory(_json(200, {"answer": "hi"}))

    auth_client.request(state, API, "POST", "/chat", json={"question": "q"}, client_factory=make)

    assert sent[0]["url"] == f"{API}/chat"
    assert sent[0]["headers"]["Authorization"] == "Bearer tok-1"


def test_an_authenticated_upload_carries_the_bearer_token() -> None:
    state = {"access_token": "tok-1", "refresh_token": "r"}
    make, sent = _factory(_json(200, {"chunks_indexed": 3}))

    auth_client.request(
        state, API, "POST", "/upload", files={"file": ("a.pdf", b"x", "application/pdf")}, client_factory=make
    )

    assert sent[0]["headers"]["Authorization"] == "Bearer tok-1"


# ---------------------------------------------------------------------------
# 3. The 401 rule: refresh once, retry once, then require a login
# ---------------------------------------------------------------------------


def test_a_401_refreshes_once_and_retries_the_original_request_once() -> None:
    state = {"access_token": "expired", "refresh_token": "r-1"}
    make, sent = _factory(
        _json(401, {"detail": "expired"}),
        _json(200, _tokens(access="access-2", refresh="refresh-2")),
        _json(200, {"answer": "hi"}),
    )

    response = auth_client.request(state, API, "POST", "/chat", json={"question": "q"}, client_factory=make)

    assert response.status_code == 200
    assert [call["url"] for call in sent] == [f"{API}/chat", f"{API}/refresh", f"{API}/chat"]
    # The retry uses the NEW token, and the rotated pair fully replaces the old.
    assert sent[2]["headers"]["Authorization"] == "Bearer access-2"
    assert state["access_token"] == "access-2"
    assert state["refresh_token"] == "refresh-2"


def test_a_failed_refresh_clears_the_tokens_and_does_not_retry() -> None:
    state = {"access_token": "expired", "refresh_token": "r-1", "user_role": "admin"}
    make, sent = _factory(_json(401, {"detail": "expired"}), _json(401, {"detail": "revoked"}))

    response = auth_client.request(state, API, "POST", "/chat", json={"question": "q"}, client_factory=make)

    assert response.status_code == 401
    assert [call["url"] for call in sent] == [f"{API}/chat", f"{API}/refresh"]
    assert auth_client.is_authenticated(state) is False
    assert state == {}


def test_the_retry_happens_at_most_once() -> None:
    """A second 401 after a successful refresh is returned, never retried
    again -- an unbounded loop here is credential stuffing against our own API."""
    state = {"access_token": "expired", "refresh_token": "r-1"}
    make, sent = _factory(
        _json(401, {"detail": "expired"}),
        _json(200, _tokens(access="access-2", refresh="refresh-2")),
        _json(401, {"detail": "still no"}),
    )

    response = auth_client.request(state, API, "POST", "/chat", json={"q": 1}, client_factory=make)

    assert response.status_code == 401
    assert len(sent) == 3


def test_a_401_with_no_refresh_token_is_returned_untouched() -> None:
    state = {"access_token": "tok"}
    make, sent = _factory(_json(401, {"detail": "nope"}))

    response = auth_client.request(state, API, "POST", "/chat", json={"q": 1}, client_factory=make)

    assert response.status_code == 401
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# 4. Public registration cannot choose a role
# ---------------------------------------------------------------------------


def test_public_registration_has_no_role_field() -> None:
    """Privileged accounts are provisioned by `scripts/create_privileged_user.py`,
    never by an unauthenticated caller asking for one."""
    assert "role" not in RegisterRequest.model_fields
    request = RegisterRequest(email="u@example.com", password="a-long-password", full_name="A User")
    assert not hasattr(request, "role")


def test_a_registration_payload_naming_admin_does_not_set_it() -> None:
    request = RegisterRequest.model_validate(
        {"email": "u@example.com", "password": "a-long-password", "full_name": "A User", "role": "admin"}
    )
    assert getattr(request, "role", None) is None


# ---------------------------------------------------------------------------
# 5. The KB review flow: role, numbered selection, confirmation, audit
# ---------------------------------------------------------------------------

_RECORDS = [
    {
        "_id": "stg-1", "original_filename": "bare_act_draft.pdf", "status": "needs_review",
        "reason": "low OCR quality", "content_hash": "hash-aaa", "ingestion_source": "admin_upload",
        "current_path": "/staging/bare_act_draft.pdf", "file_exists": True,
    },
    {
        "_id": "stg-2", "original_filename": "circular_2026.pdf", "status": "needs_review",
        "reason": "missing metadata", "content_hash": "hash-bbb", "ingestion_source": "bulk_import",
        "current_path": "/staging/circular_2026.pdf", "file_exists": True,
    },
]


def _staging(status: str | None = None) -> dict[str, Any]:
    return {"status_filter": status or "actionable", "count": len(_RECORDS), "records": list(_RECORDS)}


def _context(message: str, facts: dict[str, Any] | None = None, *, role: str = "admin") -> WorkflowContext:
    return WorkflowContext(
        session_id="s1", message=message, language="hinglish", memory={},
        facts=facts if facts is not None else {},
        authenticated_user_id="admin-user-1", claims={"sub": "admin-user-1", "role": role},
    )


def _workflow() -> AdminKnowledgeBaseWorkflow:
    return AdminKnowledgeBaseWorkflow()


def test_the_review_queue_is_admin_only() -> None:
    """The role gate is declared on the workflow and checked by the
    orchestrator before a single fact is collected."""
    assert _workflow().required_role is Role.admin


def test_a_non_admin_is_refused_before_any_fact_is_collected() -> None:
    from app.chatops.orchestrator import _role_permits

    assert _role_permits(Role.admin, "user") is False
    assert _role_permits(Role.admin, "notary") is False
    assert _role_permits(Role.admin, "admin") is True


def test_needs_review_files_dikhao_reaches_the_admin_workflow() -> None:
    match = best_match("needs review files dikhao", "hinglish")
    assert match is not None
    assert match[0].name == "admin_knowledge_base"


def test_needs_review_files_dikhao_lists_numbered_files_with_id_reason_and_source() -> None:
    workflow = _workflow()
    context = _context("needs review files dikhao", {"action": "staging_review"})
    with patch("app.services.admin_operations.staging_records", AsyncMock(side_effect=lambda s=None: _staging(s))):
        turn = _run(workflow.execute(context))

    assert "1. **bare_act_draft.pdf**" in turn.message
    assert "2. **circular_2026.pdf**" in turn.message
    assert "stg-1" in turn.message and "stg-2" in turn.message
    assert "low OCR quality" in turn.message
    assert "admin_upload" in turn.message


def test_file_2_approve_karo_resolves_the_exact_record() -> None:
    workflow = _workflow()
    context = _context("file 2 approve karo", {"action": "staging_approve"})
    with patch("app.services.admin_operations.staging_records", AsyncMock(side_effect=lambda s=None: _staging(s))):
        facts = _run(workflow.extract_facts(context))

    assert facts["staging_id"] == "stg-2"
    assert facts["staging_filename"] == "circular_2026.pdf"


def test_approval_requires_an_explicit_confirmation_showing_filename_hash_and_source() -> None:
    workflow = _workflow()
    facts = {
        "action": "staging_approve", "staging_id": "stg-2", "staging_filename": "circular_2026.pdf",
        "_review_records": {"stg-2": {
            "filename": "circular_2026.pdf", "content_hash": "hash-bbb",
            "source": "bulk_import", "status": "needs_review", "file_exists": "yes",
        }},
    }
    context = _context("file 2 approve karo", facts)

    assert workflow.needs_confirmation(context) is True
    summary = workflow.summarize_for_confirmation(context)
    assert "circular_2026.pdf" in summary
    assert "hash-bbb" in summary
    assert "bulk_import" in summary
    assert "stg-2" in summary


def test_a_confirmed_approval_calls_the_existing_kb_approve_service() -> None:
    workflow = _workflow()
    context = _context("haan", {
        "action": "staging_approve", "staging_id": "stg-2", "staging_filename": "circular_2026.pdf",
    })
    approve = AsyncMock(return_value={"status": "pending", "staging_id": "stg-2"})
    with patch("app.services.admin_operations.kb_approve", approve), \
            patch.object(type(_AuditProbe.service), "record", _AuditProbe.record):
        turn = _run(workflow.execute(context))

    approve.assert_awaited_once_with("stg-2")
    assert turn.status == "completed"


class _AuditProbe:
    """Captures what would be written to the audit log."""

    from app.services.phase3 import AuditService as _Service

    service = _Service()
    calls: ClassVar[list[dict[str, Any]]] = []

    @staticmethod
    async def record(_self: Any, **kwargs: Any) -> str:
        _AuditProbe.calls.append(kwargs)
        return "audit-1"


def test_an_approval_records_the_actor_user_id_and_the_action() -> None:
    _AuditProbe.calls.clear()
    workflow = _workflow()
    context = _context("haan", {
        "action": "staging_approve", "staging_id": "stg-1", "staging_filename": "bare_act_draft.pdf",
    })
    with patch("app.services.admin_operations.kb_approve", AsyncMock(return_value={"status": "pending"})), \
            patch.object(type(_AuditProbe.service), "record", _AuditProbe.record):
        _run(workflow.execute(context))

    (entry,) = _AuditProbe.calls
    assert entry["actor_user_id"] == "admin-user-1"
    assert entry["action"] == "kb_approve_needs_review"
    assert entry["resource_id"] == "stg-1"
    assert entry["details"]["channel"] == "chat"


def test_an_action_with_no_attributable_actor_is_refused_rather_than_recorded() -> None:
    workflow = _workflow()
    context = WorkflowContext(
        session_id="s1", message="haan", language="hinglish", memory={},
        facts={"action": "staging_approve", "staging_id": "stg-1"},
        authenticated_user_id=None, claims={"role": "admin"},
    )
    approve = AsyncMock()
    with patch("app.services.admin_operations.kb_approve", approve):
        turn = _run(workflow.execute(context))

    approve.assert_not_awaited()
    assert turn.status == "awaiting_reauth"


# ---------------------------------------------------------------------------
# 6. Rejection needs a reason; approval refuses anything not needs_review
# ---------------------------------------------------------------------------


def test_an_archive_cannot_proceed_without_a_reason() -> None:
    workflow = _workflow()
    context = _context("file 2 reject karo", {"action": "staging_archive", "staging_id": "stg-2"})

    assert "reason" in workflow.required_fields(context)
    question = workflow.next_question(context, ["reason"])
    assert "why" in question.lower() or "reason" in question.lower()


def test_a_reason_given_in_the_same_breath_is_picked_up() -> None:
    workflow = _workflow()
    context = _context("file 2 reject karo kyunki scan unreadable hai", {"action": "staging_archive"})
    with patch("app.services.admin_operations.staging_records", AsyncMock(side_effect=lambda s=None: _staging(s))):
        facts = _run(workflow.extract_facts(context))

    assert facts["staging_id"] == "stg-2"
    assert "scan unreadable" in facts["reason"]


def test_the_archive_reason_reaches_the_service_and_the_audit_log() -> None:
    _AuditProbe.calls.clear()
    workflow = _workflow()
    context = _context("haan", {
        "action": "staging_archive", "staging_id": "stg-2",
        "staging_filename": "circular_2026.pdf", "reason": "scan unreadable",
    })
    archive = AsyncMock(return_value={"status": "archived"})
    with patch("app.services.admin_operations.kb_archive", archive), \
            patch.object(type(_AuditProbe.service), "record", _AuditProbe.record):
        turn = _run(workflow.execute(context))

    archive.assert_awaited_once_with("stg-2", "scan unreadable")
    (entry,) = _AuditProbe.calls
    assert entry["action"] == "kb_archive_rejected"
    assert entry["details"]["reason"] == "scan unreadable"
    assert "scan unreadable" in turn.message


def test_only_needs_review_records_are_offered_as_candidates() -> None:
    """The status rule is enforced by the candidate set as well as by the
    service: a failed record is never in the list to be picked."""
    workflow = _workflow()
    context = _context("file 1 approve karo", {"action": "staging_approve"})
    seen: list[str | None] = []

    async def _records(status: str | None = None) -> dict[str, Any]:
        seen.append(status)
        return _staging(status)

    with patch("app.services.admin_operations.staging_records", _records):
        _run(workflow.extract_facts(context))

    assert seen == ["needs_review"]


def test_a_service_refusal_is_shown_rather_than_retried() -> None:
    """`approve_needs_review` refuses anything that is not `needs_review`
    (corrupt and failed records included). That refusal is the answer."""
    from app.core.exceptions import BadRequestError

    workflow = _workflow()
    context = _context("haan", {"action": "staging_approve", "staging_id": "stg-9"})
    approve = AsyncMock(side_effect=BadRequestError("Only needs_review documents can be approved"))
    with patch("app.services.admin_operations.kb_approve", approve):
        turn = _run(workflow.execute(context))

    approve.assert_awaited_once()
    assert turn.status == "failed"
    assert "needs_review" in turn.message


# ---------------------------------------------------------------------------
# 7. No bulk approval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["sab approve kar do", "all files approve karo", "approve everything"])
def test_a_bulk_approval_resolves_no_record_and_asks_again(message: str) -> None:
    """`selection.resolve` returns None for a message naming no single record,
    so the no-bulk rule falls out of the resolver rather than a separate check."""
    workflow = _workflow()
    context = _context(message, {"action": "staging_approve"})
    with patch("app.services.admin_operations.staging_records", AsyncMock(side_effect=lambda s=None: _staging(s))):
        facts = _run(workflow.extract_facts(context))

    assert "staging_id" not in facts
    context.facts.update(facts)
    assert "staging_id" in workflow.required_fields(context)
    assert "bulk" in workflow.next_question(context, ["staging_id"]).lower()


def test_one_approval_touches_exactly_one_record() -> None:
    _AuditProbe.calls.clear()
    workflow = _workflow()
    context = _context("haan", {
        "action": "staging_approve", "staging_id": "stg-1", "staging_filename": "bare_act_draft.pdf",
    })
    approve = AsyncMock(return_value={"status": "pending"})
    with patch("app.services.admin_operations.kb_approve", approve), \
            patch.object(type(_AuditProbe.service), "record", _AuditProbe.record):
        _run(workflow.execute(context))

    assert approve.await_count == 1
    assert len(_AuditProbe.calls) == 1
