import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.rag.kb_jurisdiction import (
    PROVENANCE_AUTOMATED_OFFICIAL,
    JurisdictionMetadataError,
    chunk_temporally_eligible,
    normalize_jurisdiction,
)


def evidence():
    return {
        "official_url": "https://example.gov.in/act.pdf", "official_sha256": "a" * 64,
        "local_sha256": "a" * 64, "content_sha256": "b" * 64,
        "policy_version": "v1", "verified_at": "2026-09-05T00:00:00+00:00",
        "identity_checks": [True], "applicability_checks": [True],
        "commencement_checks": [True], "commencement_url": "https://example.gov.in/start.pdf",
        "commencement_sha256": "c" * 64, "exact_byte_match": True,
    }


def test_machine_verified_requires_automated_provenance_and_complete_evidence():
    raw = {
        "issuing_level": "central", "applicability": "all_india",
        "source_url": "https://example.gov.in/act.pdf",
        "verification_status": "machine_verified", "machine_verification": evidence(),
    }
    assert normalize_jurisdiction(raw, provenance=PROVENANCE_AUTOMATED_OFFICIAL).metadata["review_status"] == "approved"
    with pytest.raises(JurisdictionMetadataError):
        normalize_jurisdiction(raw, provenance="manual")
    raw["machine_verification"] = {"official_url": "x"}
    assert normalize_jurisdiction(raw, provenance=PROVENANCE_AUTOMATED_OFFICIAL).metadata["review_status"] == "needs_review"


def test_machine_cannot_claim_human_verified():
    normalized = normalize_jurisdiction({
        "issuing_level": "central", "applicability": "all_india", "source_url": "https://example.gov.in",
        "verification_status": "verified",
    }, provenance=PROVENANCE_AUTOMATED_OFFICIAL)
    assert normalized.metadata["review_status"] == "needs_review"
    assert normalized.metadata["verification_status"] != "verified"


def test_explicit_not_in_force_is_excluded_even_without_query_date():
    assert chunk_temporally_eligible({"in_force": False}, None) is False
    assert chunk_temporally_eligible({"in_force": False}, "2026-09-05") is False


def test_machine_verified_answer_disclosure_is_explicit():
    from app.schemas.common import SourceCitation
    from app.services.chat_service import _machine_verification_disclosure
    source = SourceCitation(title="BSA", source_document="bsa.pdf", verification_status="machine_verified")
    assert "automatically verified" in _machine_verification_disclosure("english", [source])
    assert "human legal review" in _machine_verification_disclosure("hinglish", [source])


def test_verifier_holds_hash_mismatch_without_publishing(tmp_path, monkeypatch):
    from app.services import kb_machine_verification as module
    policy = module.POLICIES[0]
    (tmp_path / policy.filename).write_bytes(b"local")
    monkeypatch.setattr(module.settings, "knowledge_base_dir", tmp_path)
    publisher = SimpleNamespace(update_machine_verified_metadata=AsyncMock())
    fetcher = AsyncMock(side_effect=[(b"official", "application/pdf"), (b"notice", "application/pdf")])
    db = {module.UPLOADED_DOCUMENTS: SimpleNamespace(find_one=AsyncMock(return_value=None))}
    result = asyncio.run(module.MachineVerificationService(fetcher, publisher, db).verify(policy))
    assert result["status"] == "held"
    publisher.update_machine_verified_metadata.assert_not_awaited()


def test_verifier_quarantines_previously_machine_verified_source_on_mismatch(tmp_path, monkeypatch):
    from app.services import kb_machine_verification as module
    policy = module.POLICIES[0]
    (tmp_path / policy.filename).write_bytes(b"changed local")
    monkeypatch.setattr(module.settings, "knowledge_base_dir", tmp_path)
    document = {"_id": "doc", "filename": policy.filename, "metadata": {
        "verification_status": "machine_verified", "issuing_level": "central",
        "applicability": "all_india", "source_url": policy.official_url,
    }}
    db = {module.UPLOADED_DOCUMENTS: SimpleNamespace(find_one=AsyncMock(return_value=document))}
    publisher = SimpleNamespace(
        update_machine_verified_metadata=AsyncMock(), update_jurisdiction_metadata=AsyncMock(return_value={})
    )
    result = asyncio.run(module.MachineVerificationService(
        AsyncMock(side_effect=[(b"official", "application/pdf"), (b"notice", "application/pdf")]), publisher, db
    ).verify(policy))
    assert result["status"] == "held" and result["quarantined"] is True
    assert publisher.update_jurisdiction_metadata.await_args.args[1]["verification_status"] == "unverified"


def test_scheduler_does_not_start_when_disabled(monkeypatch):
    from app.services.kb_machine_verification import MachineVerificationScheduler

    monkeypatch.setattr("app.services.kb_machine_verification.settings.kb_machine_verification_enabled", False)
    scheduler = MachineVerificationScheduler(service=AsyncMock())
    scheduler.start()
    assert scheduler.task is None


def test_scheduler_starts_exactly_one_task_when_enabled(monkeypatch):
    from app.services.kb_machine_verification import MachineVerificationScheduler

    monkeypatch.setattr("app.services.kb_machine_verification.settings.kb_machine_verification_enabled", True)
    scheduler = MachineVerificationScheduler(service=SimpleNamespace(run=AsyncMock(return_value={})))

    async def start_and_stop():
        scheduler.start()
        task = scheduler.task
        assert task is not None and not task.done()
        scheduler.start()  # calling again must not spawn a second task
        assert scheduler.task is task
        await scheduler.close()
        assert scheduler.task is None

    asyncio.run(start_and_stop())


def test_scheduler_close_is_safe_when_never_started():
    from app.services.kb_machine_verification import MachineVerificationScheduler

    asyncio.run(MachineVerificationScheduler(service=AsyncMock()).close())


def test_scheduler_loop_catches_a_failed_cycle_instead_of_dying(monkeypatch):
    from app.services.kb_machine_verification import MachineVerificationScheduler

    service = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))
    scheduler = MachineVerificationScheduler(service=service)

    async def run_one_iteration():
        sleep_calls = 0

        async def fake_sleep(_seconds: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            raise asyncio.CancelledError

        monkeypatch.setattr("app.services.kb_machine_verification.asyncio.sleep", fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await scheduler._loop()
        assert sleep_calls == 1  # reached the sleep, i.e. the RuntimeError was caught, not raised

    asyncio.run(run_one_iteration())
