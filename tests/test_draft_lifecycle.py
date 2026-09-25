"""Part 57 "Drafting Lifecycle Redesign": regression tests for the new
PREVIEW_READY -> EDITING -> APPROVED -> LOCKED -> EXPORTED state machine on
`LegalDraftEngine` (approve/lock/unlock/rollback/export guards) and its
in-chat wiring (`DraftConversationEngine`).

Mocking pattern mirrors `test_drafting.py`'s existing
`test_regenerate_with_language_override_persists_language_and_sections`:
`engine.drafts`/`engine.versions` repository methods are monkeypatched
directly (no real MongoDB connection), since these are unit tests of the
engine's own guard logic, not integration tests of the repository layer.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import DraftLockedError, NotFoundError
from app.drafting.conversation import DraftConversationEngine
from app.drafting.engine import LegalDraftEngine


def _draft(**overrides: object) -> dict[str, object]:
    base = {
        "_id": "draft-1",
        "draft_type": "police_complaint",
        "template_name": "Police Complaint",
        "language": "english",
        "session_id": "s1",
        "user_id": None,
        "fields": {"applicant_name": "Ramesh Kumar"},
        "sections": {"Recipient": "content"},
        "lifecycle_state": "preview_ready",
    }
    base.update(overrides)
    return base


def _engine_with_draft(draft: dict[str, object]) -> LegalDraftEngine:
    engine = LegalDraftEngine()
    engine.drafts.find_by_id = AsyncMock(return_value=draft)
    engine.drafts.update_by_id = AsyncMock(return_value=True)
    return engine


# ---------------------------------------------------------------------------
# approve() / lock() / unlock() -- happy path and guards
# ---------------------------------------------------------------------------


def test_approve_transitions_preview_ready_to_approved() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready"))
    state = asyncio.run(engine.approve("draft-1"))
    assert state == "approved"
    engine.drafts.update_by_id.assert_awaited_once()
    _, payload = engine.drafts.update_by_id.await_args.args
    assert payload["lifecycle_state"] == "approved"


def test_approve_rejects_a_draft_that_is_already_locked() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="locked"))
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.approve("draft-1"))


def test_lock_requires_approved_state_first() -> None:
    # Directly locking a preview_ready draft (skipping approve()) must fail --
    # the spec's Approve -> "Lock this draft?" -> Yes -> Locked sequence is
    # two explicit steps, not one.
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready"))
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.lock("draft-1"))


def test_lock_succeeds_from_approved_state() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="approved"))
    state = asyncio.run(engine.lock("draft-1"))
    assert state == "locked"
    _, payload = engine.drafts.update_by_id.await_args.args
    assert payload["lifecycle_state"] == "locked"


def test_unlock_returns_a_locked_draft_to_preview_ready() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="locked"))
    state = asyncio.run(engine.unlock("draft-1"))
    assert state == "preview_ready"


def test_unlock_rejects_a_draft_that_is_already_editable() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready"))
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.unlock("draft-1"))


# ---------------------------------------------------------------------------
# regenerate() -- editing blocked once approved/locked/exported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocked_state", ["approved", "locked", "exported"])
def test_regenerate_is_blocked_once_approved_locked_or_exported(blocked_state: str) -> None:
    engine = _engine_with_draft(_draft(lifecycle_state=blocked_state))
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.regenerate("draft-1", {"applicant_name": "New Name"}))


@pytest.mark.parametrize("editable_state", ["preview_ready", "editing"])
def test_regenerate_is_allowed_while_editable(editable_state: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock as _AsyncMock

    from app.drafting.templates.base import structure_sections_for
    from app.llm.base import LLMResponse

    engine = _engine_with_draft(_draft(lifecycle_state=editable_state))
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    engine.llm.chat = _AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    engine.versions.latest_for_draft = _AsyncMock(return_value=None)
    engine.versions.insert = _AsyncMock(return_value="version-2")
    # `regenerate` now writes via `compare_and_swap` (BUG-015's optimistic
    # -concurrency check), not the plain `update_by_id` `_engine_with_draft`
    # sets up for every OTHER lifecycle method in this file.
    engine.drafts.compare_and_swap = _AsyncMock(return_value=True)

    response = asyncio.run(engine.regenerate("draft-1", {"applicant_name": "New Name"}))
    assert response.status == "complete"
    # Settles back to preview_ready regardless of which editable state it
    # started from -- "editing" is a transient chat-turn label, not a
    # separate persisted resting state (see `DraftTurnInfo.stage` docstring
    # in `app/schemas/drafting.py`).
    _, _, payload = engine.drafts.compare_and_swap.await_args.args
    assert payload["lifecycle_state"] == "preview_ready"


def test_translate_with_a_draft_id_persists_the_translated_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression for qa-40q-multilingual-20260921 BUG-08: `translate()` used
    # to run a standalone LLM pass over the draft's rendered text and return
    # it WITHOUT writing anything back -- `app/chatops/workflows/drafts.py`'s
    # "translate" action then showed that text alongside a PDF-download
    # button for the SAME draft_id, and the PDF was the untranslated
    # original (a completed draft diverging from its own export). It must
    # now persist via the same `compare_and_swap` path `regenerate()` uses,
    # so a later export of this draft_id reads the translated content.
    from unittest.mock import AsyncMock as _AsyncMock

    from app.drafting.templates.base import structure_sections_for
    from app.llm.base import LLMResponse
    from app.schemas.drafting import DraftTranslateRequest

    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready"))
    sectioned_response = "\n".join(
        f"## {heading}\nContent for {heading}." for heading in structure_sections_for("Complaint")
    )
    engine.llm.chat = _AsyncMock(return_value=LLMResponse(content=sectioned_response, model="test", provider="test"))
    engine.versions.latest_for_draft = _AsyncMock(return_value=None)
    engine.versions.insert = _AsyncMock(return_value="version-2")
    engine.drafts.compare_and_swap = _AsyncMock(return_value=True)

    result = asyncio.run(engine.translate(DraftTranslateRequest(draft_id="draft-1", target_language="hindi")))

    engine.drafts.compare_and_swap.assert_awaited_once()
    _, _, payload = engine.drafts.compare_and_swap.await_args.args
    assert payload["language"] == "hindi"
    assert payload["sections"]
    assert result.translated_text
    assert result.target_language == "hindi"


def test_translate_without_a_draft_id_still_works_as_a_scratch_conversion() -> None:
    # Ad-hoc text translation (no draft involved) has nothing to persist and
    # must keep working exactly as before.
    from unittest.mock import AsyncMock as _AsyncMock

    from app.llm.base import LLMResponse
    from app.schemas.drafting import DraftTranslateRequest

    engine = LegalDraftEngine()
    engine.llm.chat = _AsyncMock(return_value=LLMResponse(content="अनुवादित पाठ", model="test", provider="test"))

    result = asyncio.run(
        engine.translate(DraftTranslateRequest(text="Some text", target_language="hindi", source_language="english"))
    )
    assert result.translated_text == "अनुवादित पाठ"
    assert result.target_language == "hindi"


# ---------------------------------------------------------------------------
# export() -- downloadable from any live state (the approve -> lock gate that
# used to guard this was removed by product decision)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["preview_ready", "editing", "approved"])
def test_export_succeeds_from_every_pre_lock_state(state: str, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A draft is downloadable the moment it exists. This previously raised
    DraftLockedError for all three states, forcing the user through an
    "approve" -> "lock" word pair before any download button appeared."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    engine = _engine_with_draft(_draft(lifecycle_state=state, sections={"Subject": "S", "Prayer": "P"}))
    path = asyncio.run(engine.export("draft-1", "txt"))
    assert path.exists()


def test_export_from_an_editable_state_does_not_freeze_the_draft(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Downloading must not end the editing session -- only locked->exported
    is a recorded transition, so a user can download, then keep editing."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready", sections={"Subject": "S"}))
    asyncio.run(engine.export("draft-1", "txt"))
    engine.drafts.update_by_id.assert_not_awaited()


def test_export_succeeds_once_locked_and_flips_state_to_exported(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    engine = _engine_with_draft(
        _draft(
            lifecycle_state="locked",
            sections={
                "Recipient": "content",
                "Subject": "Test subject",
                "Complainant Details": "Ramesh Kumar",
                "Introduction": "Sir,",
                "Facts of the Case": "Facts.",
                "Legal Position": "Position.",
                "Consequences": "Consequences.",
                "Prayer": "Prayer.",
                "Signature Block": "Sd/-",
            },
        )
    )
    path = asyncio.run(engine.export("draft-1", "txt"))
    assert path.exists()
    _, payload = engine.drafts.update_by_id.await_args.args
    assert payload["lifecycle_state"] == "exported"


def test_export_remains_allowed_once_already_exported(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A second/third export (PDF then DOCX then TXT, per the spec's own
    # Lock -> Download PDF -> Download DOCX flow) must not be blocked just
    # because the draft already moved past "locked" into "exported".
    from app.core.config import settings

    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    engine = _engine_with_draft(
        _draft(
            lifecycle_state="exported",
            sections={
                "Recipient": "content", "Subject": "s", "Complainant Details": "c",
                "Introduction": "i", "Facts of the Case": "f", "Legal Position": "l",
                "Consequences": "co", "Prayer": "p", "Signature Block": "sig",
            },
        )
    )
    path = asyncio.run(engine.export("draft-1", "txt"))
    assert path.exists()


# ---------------------------------------------------------------------------
# Backward compatibility: a draft persisted before this change has no
# `lifecycle_state` field at all.
# ---------------------------------------------------------------------------


def test_draft_with_no_lifecycle_state_defaults_to_locked_not_editable() -> None:
    # Matches today's actual pre-existing behavior (exports were always
    # open) rather than retroactively blocking downloads on documents
    # already in flight -- see `_lifecycle_state_of`'s docstring.
    engine = _engine_with_draft(_draft())
    del engine.drafts.find_by_id.return_value["lifecycle_state"]  # type: ignore[union-attr]
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.regenerate("draft-1", {"applicant_name": "New Name"}))


def test_draft_with_no_lifecycle_state_is_exportable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "draft_output_dir", tmp_path)
    draft = _draft(
        sections={
            "Recipient": "content", "Subject": "s", "Complainant Details": "c",
            "Introduction": "i", "Facts of the Case": "f", "Legal Position": "l",
            "Consequences": "co", "Prayer": "p", "Signature Block": "sig",
        }
    )
    del draft["lifecycle_state"]
    engine = _engine_with_draft(draft)
    path = asyncio.run(engine.export("draft-1", "txt"))
    assert path.exists()


# ---------------------------------------------------------------------------
# rollback() -- content + state, always append-only
# ---------------------------------------------------------------------------


def test_rollback_restores_an_earlier_version_and_appends_a_new_one() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready", sections={"Recipient": "current content"}))
    old_version = {
        "_id": "v1", "draft_id": "draft-1", "version_number": 1,
        "sections": {"Recipient": "original content"}, "language": "english", "fields": {"applicant_name": "Old Name"},
    }
    engine.versions.get_version = AsyncMock(return_value=old_version)
    engine.versions.latest_for_draft = AsyncMock(return_value={"_id": "v2", "version_number": 2})
    engine.versions.insert = AsyncMock(return_value="v3")
    engine.versions.update_by_id = AsyncMock(return_value=True)

    response = asyncio.run(engine.rollback("draft-1", 1))

    assert response.sections["Recipient"] == "original content"
    # The rollback itself is recorded as a NEW version (3), not a mutation
    # of version 1 or 2 -- "preserve version history" per the spec.
    engine.versions.insert.assert_awaited_once()
    inserted_payload = engine.versions.insert.await_args.args[0]
    assert inserted_payload["sections"]["Recipient"] == "original content"
    assert "Rolled back to version 1" in inserted_payload["note"]
    # Old version 2 is marked superseded, never deleted.
    engine.versions.update_by_id.assert_awaited_once()


def test_rollback_is_blocked_on_a_locked_draft() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="locked"))
    with pytest.raises(DraftLockedError):
        asyncio.run(engine.rollback("draft-1", 1))


def test_rollback_to_a_nonexistent_version_raises_not_found() -> None:
    engine = _engine_with_draft(_draft(lifecycle_state="preview_ready"))
    engine.versions.get_version = AsyncMock(return_value=None)
    with pytest.raises(NotFoundError):
        asyncio.run(engine.rollback("draft-1", 99))


# ---------------------------------------------------------------------------
# Chat-layer wiring: DraftConversationEngine's approve -> lock -> export gate
# ---------------------------------------------------------------------------


def test_conversation_engine_offers_downloads_straight_from_preview() -> None:
    """"download pdf" in preview hands over the file instead of replying
    "approve it first"."""
    engine = DraftConversationEngine()
    memory: dict[str, object] = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "police_complaint",
        "draft_fields": {}, "draft_id": "draft-1", "draft_language": "english",
    }
    result = asyncio.run(engine.handle_turn("s1", "download pdf", "english", memory))
    assert result is not None
    assert result.info.available_export_formats == ["pdf", "docx", "txt", "rtf"]
    assert result.info.stage == "preview"
    assert "approve" not in result.reply_text.lower()


def test_conversation_engine_approve_is_a_no_op_that_keeps_the_draft_editable() -> None:
    """"approve" still parses (users who learned the old flow keep working)
    but no longer gates downloads or freezes the draft."""
    engine = DraftConversationEngine()
    engine.draft_engine.drafts.find_by_id = AsyncMock(return_value=_draft(lifecycle_state="preview_ready"))
    engine.draft_engine.drafts.update_by_id = AsyncMock(return_value=True)
    memory: dict[str, object] = {
        "draft_mode": True, "draft_stage": "preview", "draft_template_id": "police_complaint",
        "draft_fields": {}, "draft_id": "draft-1", "draft_language": "english",
    }
    result = asyncio.run(engine.handle_turn("s1", "Looks good, approved.", "english", memory))
    assert result is not None
    assert memory["draft_stage"] == "preview"
    assert result.info.available_export_formats == ["pdf", "docx", "txt", "rtf"]


def test_conversation_engine_lock_confirmation_populates_export_formats() -> None:
    engine = DraftConversationEngine()
    engine.draft_engine.drafts.find_by_id = AsyncMock(return_value=_draft(lifecycle_state="approved"))
    engine.draft_engine.drafts.update_by_id = AsyncMock(return_value=True)
    memory: dict[str, object] = {
        "draft_mode": True, "draft_stage": "approved", "draft_template_id": "police_complaint",
        "draft_fields": {}, "draft_id": "draft-1", "draft_language": "english",
    }
    result = asyncio.run(engine.handle_turn("s1", "yes", "english", memory))
    assert result is not None
    assert memory["draft_stage"] == "locked"
    assert set(result.info.available_export_formats) == {"pdf", "docx", "txt", "rtf"}
