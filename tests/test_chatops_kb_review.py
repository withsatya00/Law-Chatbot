import asyncio
from typing import Any

import pytest

import app.chatops.orchestrator  # noqa: F401 - registers workflows
from app.chatops.base import WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator
from app.chatops.workflows import admin

_ADMIN = {"authenticated_user_id": "admin-1", "claims": {"sub": "admin-1", "role": "admin"}}
_USER = {"authenticated_user_id": "user-1", "claims": {"sub": "user-1", "role": "user"}}

_REVIEW = [
    {
        "_id": "review-1",
        "original_filename": "consumer-act.pdf",
        "status": "needs_review",
        "content_hash": "hash-one",
        "ingestion_source": "admin_upload",
        "reason": "Readable previous failure",
        "current_path": "review/consumer-act.pdf",
        "file_exists": True,
    },
    {
        "_id": "review-2",
        "original_filename": "rti-act.pdf",
        "status": "needs_review",
        "content_hash": "hash-two",
        "ingestion_source": "official_source",
        "reason": "Needs authority verification",
        "current_path": "review/rti-act.pdf",
        "file_exists": True,
    },
]


def _turn(message: str, memory: dict[str, Any], identity: dict[str, Any]) -> WorkflowTurn | None:
    return asyncio.run(
        ChatOrchestrator().handle_turn(
            session_id="session-1", message=message, language="hinglish", memory=memory, **identity
        )
    )


@pytest.fixture
def review_services(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"approved": [], "archived": [], "closed": [], "audited": []}

    async def staging_records(status: str | None = None) -> dict[str, Any]:
        if status == "needs_review":
            records = _REVIEW
        elif status == "all":
            records = [
                *_REVIEW,
                {
                    "_id": "missing-1",
                    "original_filename": "missing-act.pdf",
                    "status": "needs_review",
                    "path_missing": True,
                    "reason": "Recorded file not found",
                    "file_exists": False,
                },
            ]
        else:
            records = []
        return {"status_filter": status or "actionable", "count": len(records), "records": records}

    async def approve(staging_id: str) -> dict[str, Any]:
        calls["approved"].append(staging_id)
        return {"staging_id": staging_id, "status": "pending"}

    async def archive(staging_id: str, reason: str | None = None) -> dict[str, Any]:
        calls["archived"].append((staging_id, reason))
        return {"staging_id": staging_id, "status": "archived"}

    async def close(staging_id: str, reason: str) -> dict[str, Any]:
        calls["closed"].append((staging_id, reason))
        return {"staging_id": staging_id, "resolution": "closed_missing_file"}

    async def record(_self: Any, **kwargs: Any) -> None:
        calls["audited"].append(kwargs)

    monkeypatch.setattr(admin.admin_operations, "staging_records", staging_records)
    monkeypatch.setattr(admin.admin_operations, "kb_approve", approve)
    monkeypatch.setattr(admin.admin_operations, "kb_archive", archive)
    monkeypatch.setattr(admin.admin_operations, "kb_close_missing", close)
    monkeypatch.setattr(admin.AuditService, "record", record)
    return calls


def test_review_queue_is_numbered_and_contains_verification_details(review_services: dict[str, list[Any]]) -> None:
    result = _turn("needs review files dikhao", {}, _ADMIN)

    assert result is not None and result.status == "completed"
    assert "1." in result.message and "2." in result.message
    assert "review-1" in result.message and "official_source" in result.message
    assert review_services["approved"] == []


def test_file_two_requires_confirmation_then_approves_and_audits_actor(
    review_services: dict[str, list[Any]],
) -> None:
    memory: dict[str, Any] = {}
    confirmation = _turn("file 2 approve karo", memory, _ADMIN)

    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert "rti-act.pdf" in confirmation.message and "hash-two" in confirmation.message
    assert review_services["approved"] == []

    done = _turn("yes", memory, _ADMIN)
    assert done is not None and done.status == "completed"
    assert review_services["approved"] == ["review-2"]
    assert review_services["audited"][0]["actor_user_id"] == "admin-1"
    assert review_services["audited"][0]["action"] == "kb_approve_needs_review"


def test_declining_confirmation_changes_nothing(review_services: dict[str, list[Any]]) -> None:
    memory: dict[str, Any] = {}
    _turn("file 1 approve karo", memory, _ADMIN)
    result = _turn("no", memory, _ADMIN)

    assert result is not None and result.status == "cancelled"
    assert review_services["approved"] == []
    assert review_services["audited"] == []


def test_bulk_approval_is_refused_by_asking_for_one_file(review_services: dict[str, list[Any]]) -> None:
    result = _turn("approve all review files", {}, _ADMIN)

    assert result is not None and result.missing_field == "staging_id"
    assert "one at a time" in result.message.lower()
    assert review_services["approved"] == []


def test_archive_requires_reason_and_confirmation(review_services: dict[str, list[Any]]) -> None:
    memory: dict[str, Any] = {}
    reason_question = _turn("file 1 reject karo", memory, _ADMIN)
    assert reason_question is not None and reason_question.missing_field == "reason"

    confirmation = _turn("because issuing authority could not be verified", memory, _ADMIN)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert review_services["archived"] == []

    _turn("yes", memory, _ADMIN)
    assert review_services["archived"] == [
        ("review-1", "issuing authority could not be verified")
    ]


def test_non_admin_never_reaches_review_services(review_services: dict[str, list[Any]]) -> None:
    result = _turn("file 1 approve karo", {}, _USER)

    assert result is not None and result.status == "forbidden"
    assert review_services["approved"] == []
    assert review_services["audited"] == []


def test_missing_path_close_requires_reason_confirmation_and_actor(
    review_services: dict[str, list[Any]],
) -> None:
    memory: dict[str, Any] = {}
    confirmation = _turn(
        "close missing file 1 because the official replacement cannot be located",
        memory,
        _ADMIN,
    )

    assert confirmation is not None and confirmation.status == "awaiting_confirmation"
    assert review_services["closed"] == []

    _turn("yes", memory, _ADMIN)
    assert review_services["closed"] == [
        ("missing-1", "the official replacement cannot be located")
    ]
    assert review_services["audited"][0]["actor_user_id"] == "admin-1"
    assert review_services["audited"][0]["action"] == "kb_close_missing_file"


def test_read_only_status_request_replaces_an_unconfirmed_reindex(
    review_services: dict[str, list[Any]],
) -> None:
    memory: dict[str, Any] = {}
    confirmation = _turn("re-index", memory, _ADMIN)
    assert confirmation is not None and confirmation.status == "awaiting_confirmation"

    status = _turn("KB staging status dikhao", memory, _ADMIN)

    assert status is not None and status.status == "completed"
    assert "staging ledger" in status.message.lower()
    assert "clear yes or no" not in status.message
