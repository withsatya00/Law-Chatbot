import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.schemas.chat import ChatRequest
from app.schemas.phase3 import (
    FollowUpRequest,
    LegalSourceMetadata,
    UserPreferencesRequest,
)
from app.services.phase3 import (
    AuditService,
    BackgroundJobService,
    EvaluationService,
    FormWorkflowService,
    LegalUpdateService,
    PreferenceService,
    next_follow_up,
)


def test_audit_record_degrades_to_a_logged_warning_when_the_write_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirmed live: `DocumentService.analyze` broke outright (a real
    request failing with a 500) the moment `AuditService.record` had no
    Mongo connection to write through -- an audit-log write is supposed to
    be a side effect of the real action, never a precondition for it.
    Matches `AuditService.event`'s own established "telemetry must never
    mask the real outcome" reasoning one method above this one.
    """
    service = AuditService()

    async def _boom(_document: dict[str, object]) -> str:
        raise RuntimeError("MongoDB client is not connected.")

    monkeypatch.setattr(service.audit, "insert", _boom)

    # Must not raise.
    asyncio.run(
        service.record(actor_user_id="user-A", action="document_accessed", resource_type="uploaded_document", resource_id="doc-1")
    )


def test_plain_language_modes_are_explicit_and_backward_compatible() -> None:
    assert ChatRequest(question="Explain bail").explanation_mode is None
    assert ChatRequest(question="Explain bail", explanation_mode="simple").explanation_mode == "simple"
    assert ChatRequest(question="Explain bail", explanation_mode="detailed").explanation_mode == "detailed"
    assert ChatRequest(question="Explain bail", explanation_mode="advocate").explanation_mode == "advocate"


@pytest.mark.parametrize(
    ("code", "section", "current_code", "current_section"),
    [("IPC", "420", "BNS", "318(4)"), ("CrPC", "154", "BNSS", "173")],
)
def test_curated_legacy_mapping_never_claims_unqualified_equivalence(
    code: str, section: str, current_code: str, current_section: str
) -> None:
    result = LegalUpdateService().map_legacy(code, section)
    assert result.current_code == current_code
    assert result.current_section == current_section
    assert result.requires_legal_verification is True
    assert "verify" in result.note.casefold()


def test_unknown_legacy_mapping_refuses_to_guess() -> None:
    result = LegalUpdateService().map_legacy("IPC", "999")
    assert result.current_code is None
    assert "do not infer" in result.note.casefold()


def test_follow_up_asks_only_first_relevant_missing_question() -> None:
    result = next_follow_up(FollowUpRequest(
        workflow="cyber_fraud",
        messages=[{"role": "user", "content": "PhonePe UPI fraud at 10:30 yesterday"}],
        known_fields={"bank": "HDFC"},
    ))
    assert result.field == "transaction_id"
    assert result.question and "UTR" in result.question


def test_follow_up_uses_document_facts_and_never_repeats_asked_field() -> None:
    result = next_follow_up(FollowUpRequest(
        workflow="police_complaint",
        document_texts=["Incident location: Pune. Evidence: CCTV video."],
        known_fields={"accused": "unknown"},
        asked_fields=["witnesses"],
    ))
    assert result.question is None
    assert result.remaining_fields == ["witnesses"]


@pytest.mark.parametrize(
    ("language", "expected"),
    [("english", "When did"), ("hindi", "लेन-देन"), ("hinglish", "Transaction kab")],
)
def test_follow_up_is_localized_in_required_phase3_languages(language: str, expected: str) -> None:
    result = next_follow_up(FollowUpRequest(workflow="cyber_fraud", language=language))
    assert result.field == "transaction_time"
    assert result.question and expected in result.question


@pytest.mark.asyncio
async def test_preferences_are_owner_scoped_and_deletable() -> None:
    service = PreferenceService()
    state = {}
    service.preferences.get_for_owner = AsyncMock(side_effect=lambda owner: state.get(owner))

    async def upsert(owner: str, values: dict) -> None:
        state[owner] = {"owner_user_id": owner, **values, "updated_at": datetime.now(UTC)}

    service.preferences.upsert_for_owner = AsyncMock(side_effect=upsert)
    result = await service.update("user-a", UserPreferencesRequest(language="hindi", explanation_mode="detailed"))
    assert result.language == "hindi"
    assert result.explanation_mode == "detailed"
    assert "user-b" not in state


@pytest.mark.asyncio
async def test_profile_state_code_is_normalized_and_persisted() -> None:
    """Jurisdiction Routing (Phase 2): a full State name is accepted and
    stored as the canonical code (`app.rag.kb_jurisdiction.normalize_state_code`),
    same as any other jurisdiction-metadata input in this codebase."""
    service = PreferenceService()
    state = {}
    service.preferences.get_for_owner = AsyncMock(side_effect=lambda owner: state.get(owner))
    service.preferences.upsert_for_owner = AsyncMock(
        side_effect=lambda owner, values: state.__setitem__(owner, {"owner_user_id": owner, **values})
    )
    result = await service.update("user-a", UserPreferencesRequest(profile_state_code="Madhya Pradesh"))
    assert result.profile_state_code == "MP"


@pytest.mark.asyncio
async def test_invalid_profile_state_code_is_rejected() -> None:
    service = PreferenceService()
    with pytest.raises(BadRequestError):
        await service.update("user-a", UserPreferencesRequest(profile_state_code="Narnia"))


def _form_item(owner: str = "user-a") -> dict:
    now = datetime.now(UTC)
    return {
        "_id": "form-1", "owner_user_id": owner, "form_type": "rti", "fields": {},
        "confirmed": False, "created_at": now, "updated_at": now,
    }


@pytest.mark.asyncio
async def test_form_prefill_requires_review_and_explicit_confirmation() -> None:
    service = FormWorkflowService()
    state = _form_item()
    state["fields"] = {
        "applicant_name": "Asha", "applicant_address": "Pune",
        "public_authority": "Municipal Corporation", "information_requested": "Building permission record",
    }
    service.forms.find_by_id = AsyncMock(return_value=state)
    service.forms.update_by_id = AsyncMock(return_value=True)

    with pytest.raises(BadRequestError):
        await service.confirm("user-a", "form-1", "yes")
    confirmed = await service.confirm("user-a", "form-1", "I CONFIRM THE REVIEWED FACTS")
    assert confirmed.confirmed is True
    assert confirmed.external_submission is False
    assert confirmed.review_required is True


@pytest.mark.asyncio
async def test_form_workflow_rejects_cross_owner_access() -> None:
    service = FormWorkflowService()
    service.forms.find_by_id = AsyncMock(return_value=_form_item("user-a"))
    with pytest.raises(ForbiddenError):
        await service.update("user-b", "form-1", {"applicant_name": "Other"})


def test_legal_source_tracker_flags_stale_and_superseded_sources() -> None:
    service = LegalUpdateService()
    now = datetime.now(UTC)
    item = {
        "_id": "source-1", "owner_user_id": "admin", "title": "Old Act copy",
        "source_url": "https://example.gov.in/act", "jurisdiction": "India", "act_name": "Example Act",
        "section_number": "1", "source_version": "2020", "effective_date": None,
        "status": "superseded", "amendment_notes": "Replaced", "last_verified_date": date.today() - timedelta(days=200),
        "update_date": None, "verification_status": "verified", "created_at": now, "updated_at": now,
    }
    response = service._to_response(item)
    assert response.stale is True
    assert response.current_as_of == item["last_verified_date"].isoformat()


@pytest.mark.asyncio
async def test_legal_source_dates_are_bson_safe() -> None:
    service = LegalUpdateService()
    service.sources.insert = AsyncMock(return_value="source-1")
    result = await service.create(
        "admin-a",
        LegalSourceMetadata(title="Current Act", last_verified_date=date.today(), effective_date=date(2024, 7, 1)),
    )
    inserted = service.sources.insert.await_args.args[0]
    assert isinstance(inserted["last_verified_date"], datetime)
    assert isinstance(inserted["effective_date"], datetime)
    assert result.last_verified_date == date.today()


@pytest.mark.asyncio
async def test_background_jobs_are_owner_scoped() -> None:
    service = BackgroundJobService()
    now = datetime.now(UTC)
    service.jobs.find_by_id = AsyncMock(return_value={
        "_id": "job-1", "owner_user_id": "user-a", "job_type": "export", "status": "queued",
        "progress": 0, "result": {}, "created_at": now, "updated_at": now,
    })
    with pytest.raises(ForbiddenError):
        await service.get("user-b", "job-1")


@pytest.mark.asyncio
async def test_only_failed_background_jobs_can_be_retried_by_owner() -> None:
    service = BackgroundJobService()
    now = datetime.now(UTC)
    item = {
        "_id": "job-1", "owner_user_id": "user-a", "job_type": "ocr", "status": "failed",
        "progress": 5, "error": "temporary failure", "result": {}, "created_at": now, "updated_at": now,
    }
    service.jobs.find_by_id = AsyncMock(return_value=item)
    service.jobs.update_by_id = AsyncMock(return_value=True)
    retried = await service.retry("user-a", "job-1")
    assert retried.status == "queued"
    assert retried.progress == 0
    service.jobs.update_by_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_multilingual_evaluation_benchmark_scores_all_dimensions() -> None:
    service = EvaluationService()
    service.runs.insert = AsyncMock(return_value="eval-1")
    result = await service.run()
    assert result.cases == 20
    assert set(result.scores) == {
        "routing_accuracy", "language_accuracy", "ordinary_question_safety",
    }
    assert all(score == 1.0 for score in result.scores.values())
    assert result.failures == []


# ---------------------------------------------------------------------------
# G6 -- historical evaluation runs must remain accessible after the run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_returns_stored_historical_evaluations() -> None:
    """Security/correctness finding G6: `POST /evaluations/run`'s response
    was previously the ONLY way to ever see a run's result."""
    service = EvaluationService()
    stored = {
        "_id": "eval-1", "status": "complete",
        "scores": {"routing_accuracy": 0.9, "language_accuracy": 1.0, "ordinary_question_safety": 1.0},
        "case_count": 20, "failures": [], "latency_ms": 123.4,
    }
    service.runs.list_recent = AsyncMock(return_value=[stored])

    results = await service.list_runs()

    assert len(results) == 1
    assert results[0].run_id == "eval-1"
    assert results[0].cases == 20
    assert results[0].scores["routing_accuracy"] == 0.9


@pytest.mark.asyncio
async def test_get_run_returns_a_specific_historical_evaluation() -> None:
    service = EvaluationService()
    stored = {
        "_id": "eval-2", "status": "complete", "scores": {}, "case_count": 5,
        "failures": [{"detail": "x"}], "latency_ms": 50.0,
    }
    service.runs.find_by_id = AsyncMock(return_value=stored)

    result = await service.get_run("eval-2")

    assert result.run_id == "eval-2"
    assert result.cases == 5
    assert result.failures == [{"detail": "x"}]


@pytest.mark.asyncio
async def test_get_run_for_a_nonexistent_run_id_is_not_found() -> None:
    service = EvaluationService()
    service.runs.find_by_id = AsyncMock(return_value=None)

    with pytest.raises(NotFoundError):
        await service.get_run("no-such-run")
