"""Phase 2 Milestone B: legal-source governance.

The rule this module defends: **the platform may only present a source as
verified law when a human said so and left evidence.** Everything else here is
in service of that -- who may flip the state (admins only), what they must
supply (evidence URL + notes + a non-future date), what gets recorded (an audit
entry naming the actor), and what a migration is allowed to assume (nothing).

Exercised against in-memory repository doubles, so the governance invariants are
tested independently of MongoDB availability, matching `test_notarization.py`.
"""

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from app.core.exceptions import BadRequestError, NotFoundError
from app.schemas.phase3 import (
    LegalSourceMetadata,
    SourceReviewRequest,
    SupersedeSourceRequest,
    VerifySourceRequest,
)
from app.services.phase3 import LegalUpdateService

ADMIN = "admin-1"


class _FakeSources:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def insert(self, document: dict[str, Any]) -> str:
        document.setdefault("_id", str(uuid4()))
        self.rows[str(document["_id"])] = document
        return str(document["_id"])

    async def find_by_id(self, item_id: str) -> dict[str, Any] | None:
        return self.rows.get(item_id)

    async def update_by_id(self, item_id: str, updates: dict[str, Any]) -> bool:
        if item_id not in self.rows:
            return False
        self.rows[item_id].update(updates)
        return True

    async def list_sources(self, query: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        return [
            row for row in self.rows.values()
            if all(row.get(key) == value for key, value in query.items())
        ][:limit]


def _service() -> LegalUpdateService:
    service = LegalUpdateService.__new__(LegalUpdateService)
    service.sources = _FakeSources()
    return service


def _metadata(**overrides: Any) -> LegalSourceMetadata:
    base: dict[str, Any] = {
        "title": "Bharatiya Nyaya Sanhita, 2023",
        "act_name": "Bharatiya Nyaya Sanhita",
        "jurisdiction": "India",
    }
    base.update(overrides)
    return LegalSourceMetadata(**base)


def _created(service: LegalUpdateService, **overrides: Any) -> Any:
    return asyncio.run(service.create(ADMIN, _metadata(**overrides)))


# ---------------------------------------------------------------------------
# Default posture
# ---------------------------------------------------------------------------


def test_a_new_source_is_unverified_by_default() -> None:
    """A record nobody has reviewed must not read as one that passed review."""
    assert LegalSourceMetadata(title="x").verification_status == "unverified"
    assert _created(_service()).verification_status == "unverified"


def test_a_source_cannot_be_created_already_verified() -> None:
    """`create` is reachable from an admin upload form. Honouring a
    caller-supplied `verified` would record a human review that never happened.
    """
    source = _created(_service(), verification_status="verified")
    assert source.verification_status == "pending_review"


def test_an_official_looking_filename_confers_nothing() -> None:
    source = _created(
        _service(),
        title="bns_2023_official_gazette_of_india.pdf",
        issuing_authority="Ministry of Law and Justice",
        source_url="https://www.indiacode.nic.in/",
    )
    assert source.verification_status == "unverified"
    assert source.status == "unknown"
    assert source.last_verified_date is None


def test_created_source_invents_no_dates_or_evidence() -> None:
    source = _created(_service())
    assert source.publication_date is None
    assert source.effective_date is None
    assert source.last_verified_date is None
    assert source.evidence_url == ""
    assert source.reviewed_by is None
    assert source.current_as_of == "Not yet verified"
    assert source.stale is True


# ---------------------------------------------------------------------------
# Verification requires evidence
# ---------------------------------------------------------------------------


def test_verification_requires_an_evidence_url() -> None:
    service = _service()
    source = _created(service)
    with pytest.raises(BadRequestError):
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="verified",
            last_verified_date=date(2026, 1, 15),
            review_notes="Compared against the gazette text.",
        )))
    assert service.sources.rows[source.source_id]["verification_status"] == "unverified"


def test_verification_requires_review_notes() -> None:
    service = _service()
    source = _created(service)
    with pytest.raises(BadRequestError):
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="verified",
            last_verified_date=date(2026, 1, 15),
            evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
        )))
    assert service.sources.rows[source.source_id]["verification_status"] == "unverified"


def test_a_review_date_cannot_be_in_the_future() -> None:
    """A verification dated tomorrow describes a comparison nobody has made."""
    service = _service()
    source = _created(service)
    with pytest.raises(BadRequestError):
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="verified",
            last_verified_date=date.today() + timedelta(days=1),
            evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
            review_notes="Checked.",
        )))


def test_a_complete_review_verifies_and_records_the_reviewer() -> None:
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 15),
        evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
        review_notes="Section text compared line by line against the gazette PDF.",
        status="in_force",
    )))
    assert result.verification_status == "verified"
    assert result.status == "in_force"
    assert result.reviewed_by == ADMIN
    assert result.reviewed_at is not None
    assert result.evidence_url.startswith("https://")
    assert result.review_notes


def test_rejecting_a_source_needs_no_evidence_url() -> None:
    """Only `verified` asserts a fact about the outside world. Requiring
    evidence to REJECT would discourage recording that a source is unusable."""
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="rejected",
        last_verified_date=date(2026, 1, 15),
        review_notes="Unofficial reprint; provenance unclear.",
    )))
    assert result.verification_status == "rejected"


def test_reviewing_an_unknown_source_is_not_found() -> None:
    service = _service()
    with pytest.raises(NotFoundError):
        asyncio.run(service.review("nope", ADMIN, SourceReviewRequest(
            verification_status="rejected", last_verified_date=date(2026, 1, 15),
        )))


# ---------------------------------------------------------------------------
# Supersession lineage
# ---------------------------------------------------------------------------


def test_supersede_records_lineage_in_both_directions() -> None:
    service = _service()
    old = _created(service, title="Indian Penal Code, 1860", act_name="Indian Penal Code")
    new = _created(service, title="Bharatiya Nyaya Sanhita, 2023")

    result = asyncio.run(service.supersede(old.source_id, ADMIN, SupersedeSourceRequest(
        superseded_by_source_id=new.source_id,
        effective_date=date(2024, 7, 1),
        review_notes="BNS replaced the IPC with effect from 1 July 2024.",
    )))

    assert result.status == "superseded"
    assert result.superseded_by == new.source_id
    assert service.sources.rows[new.source_id]["supersedes"] == [old.source_id]


def test_supersede_is_idempotent() -> None:
    service = _service()
    old = _created(service)
    new = _created(service)
    request = SupersedeSourceRequest(superseded_by_source_id=new.source_id)
    asyncio.run(service.supersede(old.source_id, ADMIN, request))
    asyncio.run(service.supersede(old.source_id, ADMIN, request))
    assert service.sources.rows[new.source_id]["supersedes"] == [old.source_id]


def test_a_source_cannot_supersede_itself() -> None:
    service = _service()
    source = _created(service)
    with pytest.raises(BadRequestError):
        asyncio.run(service.supersede(source.source_id, ADMIN, SupersedeSourceRequest(
            superseded_by_source_id=source.source_id,
        )))


def test_the_superseding_source_must_be_registered() -> None:
    """Otherwise `superseded_by` is an amendment claim with no source behind it."""
    service = _service()
    source = _created(service)
    with pytest.raises(NotFoundError):
        asyncio.run(service.supersede(source.source_id, ADMIN, SupersedeSourceRequest(
            superseded_by_source_id="not-in-the-registry",
        )))


# ---------------------------------------------------------------------------
# RBAC: the routes, not just the service
# ---------------------------------------------------------------------------


def test_every_governance_route_is_admin_only() -> None:
    """The review/supersede/verify endpoints must sit behind `require_admin`.

    Checked on the router rather than by mocking a request, so adding a new
    governance route to a DIFFERENT, unguarded router fails this test.
    """
    from app.api.admin_phase3 import router
    from app.api.deps import require_admin

    guards = {getattr(dep.dependency, "__name__", "") for dep in router.dependencies}
    assert require_admin.__name__ in guards

    governed = {
        route.path for route in router.routes  # type: ignore[attr-defined]
        if "legal-sources" in getattr(route, "path", "")
    }
    for suffix in ("review", "supersede", "verify"):
        assert any(path.endswith(suffix) for path in governed), f"missing governance route: {suffix}"


def test_governance_routes_are_not_reachable_from_a_user_router() -> None:
    """A non-admin router must not expose any legal-source mutation."""
    from app.api.phase3 import router as user_router

    for route in user_router.routes:  # type: ignore[attr-defined]
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if "legal-source" in path or "legal-sources" in path:
            assert methods <= {"GET", "HEAD"}, f"user router exposes a mutation on {path}"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_defaults_never_grant_verification() -> None:
    """The backfill fills structural gaps only. No migration can perform the
    human comparison that `verified` asserts."""
    from scripts.migrate_phase2_governance import _DEFAULTS

    assert "verification_status" not in _DEFAULTS
    for key in ("last_verified_date", "effective_date", "source_url", "official_url"):
        assert key not in _DEFAULTS, f"{key} must never be invented by a migration"
    assert _DEFAULTS["evidence_url"] == ""
    assert _DEFAULTS["reviewed_by"] is None


def test_link_document_records_the_governed_chunk_ids() -> None:
    """Governance has to be traceable to the exact retrievable text it covers."""
    service = _service()
    source = _created(service)

    class _FakeCollection:
        async def count_documents(self, query: dict[str, Any]) -> int:
            return 2

        def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None) -> Any:
            async def _gen() -> Any:
                for chunk_id in ("chunk-a", "chunk-b"):
                    yield {"_id": chunk_id}

            return _gen()

        async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> None:
            return None

    class _FakeEmbeddings:
        collection = _FakeCollection()

    import app.repositories.documents as documents_module

    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = _FakeEmbeddings  # type: ignore[misc,assignment]
    try:
        result = asyncio.run(service.link_document(source.source_id, "doc-1"))
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]

    assert result.document_id == "doc-1"
    assert result.chunk_ids == ["chunk-a", "chunk-b"]


def test_review_updates_do_not_leak_reviewer_pii() -> None:
    """The audit trail records an account id, never an email address."""
    service = _service()
    source = _created(service)
    asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 15),
        evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
        review_notes="Verified against the gazette.",
    )))
    stored = service.sources.rows[source.source_id]
    assert stored["reviewed_by"] == ADMIN
    assert "@" not in str(stored["reviewed_by"])
    assert isinstance(stored["reviewed_at"], datetime)
    assert stored["reviewed_at"].tzinfo is UTC or stored["reviewed_at"].utcoffset() is not None


# ---------------------------------------------------------------------------
# G1 -- verification state changes must propagate to already-linked chunks
# ---------------------------------------------------------------------------


class _RecordingCollection:
    """Records every `update_many` call instead of just accepting and
    discarding it, so a test can assert exactly what governance metadata
    reached the chunks -- `test_link_document_records_the_governed_chunk_ids`'s
    `_FakeCollection` above only needed to not crash; these tests need to see
    the propagated values themselves."""

    def __init__(self, chunk_ids: tuple[str, ...] = ("chunk-a", "chunk-b")) -> None:
        self._chunk_ids = chunk_ids
        self.update_many_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def count_documents(self, query: dict[str, Any]) -> int:
        return len(self._chunk_ids)

    def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None) -> Any:
        async def _gen() -> Any:
            for chunk_id in self._chunk_ids:
                yield {"_id": chunk_id}

        return _gen()

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        self.update_many_calls.append((query, update))

        class _Result:
            modified_count = len(self._chunk_ids)

        return _Result()


class _RecordingEmbeddings:
    def __init__(self, chunk_ids: tuple[str, ...] = ("chunk-a", "chunk-b")) -> None:
        self.collection = _RecordingCollection(chunk_ids)


def _linked_source_with_recording_embeddings(service: LegalUpdateService) -> tuple[Any, _RecordingCollection]:
    """Creates a source and links it to `doc-1`, returning the source and the
    fake chunk collection so a caller can inspect propagation calls made
    AFTER this point (the link call itself also propagates once)."""
    source = _created(service)
    embeddings = _RecordingEmbeddings()
    import app.repositories.documents as documents_module

    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = lambda: embeddings  # type: ignore[misc,assignment]
    try:
        asyncio.run(service.link_document(source.source_id, "doc-1"))
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]
    return source, embeddings.collection


def test_verifying_a_linked_source_propagates_to_its_chunks() -> None:
    service = _service()
    source, collection = _linked_source_with_recording_embeddings(service)
    collection.update_many_calls.clear()  # only care about propagation from THIS review, not the link

    import app.repositories.documents as documents_module

    embeddings = _RecordingEmbeddings(("chunk-a", "chunk-b"))
    embeddings.collection = collection
    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = lambda: embeddings  # type: ignore[misc,assignment]
    try:
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="verified",
            last_verified_date=date(2026, 1, 15),
            evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
            review_notes="Compared line by line against the gazette PDF.",
        )))
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]

    assert len(collection.update_many_calls) == 1, "review() must re-propagate governance metadata to linked chunks"
    query, update = collection.update_many_calls[0]
    assert query == {"document_id": "doc-1"}
    assert update["$set"]["metadata.verification_status"] == "verified"


def test_rejecting_a_linked_source_propagates_rejection_to_its_chunks() -> None:
    """The exact reported symptom: a source moved to a DIFFERENT status than
    the one it was linked under must not leave its chunks presenting the
    stale, previously-linked state."""
    service = _service()
    source, collection = _linked_source_with_recording_embeddings(service)
    collection.update_many_calls.clear()

    import app.repositories.documents as documents_module

    embeddings = _RecordingEmbeddings()
    embeddings.collection = collection
    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = lambda: embeddings  # type: ignore[misc,assignment]
    try:
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="rejected",
            last_verified_date=date(2026, 1, 15),
            review_notes="Unofficial reprint discovered after initial link.",
        )))
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]

    assert len(collection.update_many_calls) == 1
    _, update = collection.update_many_calls[0]
    assert update["$set"]["metadata.verification_status"] == "rejected"


def test_superseding_a_linked_source_propagates_to_its_chunks() -> None:
    service = _service()
    source, collection = _linked_source_with_recording_embeddings(service)
    new = _created(service, title="Bharatiya Nyaya Sanhita, 2023")
    collection.update_many_calls.clear()

    import app.repositories.documents as documents_module

    embeddings = _RecordingEmbeddings()
    embeddings.collection = collection
    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = lambda: embeddings  # type: ignore[misc,assignment]
    try:
        asyncio.run(service.supersede(source.source_id, ADMIN, SupersedeSourceRequest(
            superseded_by_source_id=new.source_id,
            review_notes="Replaced by the newly registered source.",
        )))
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]

    assert len(collection.update_many_calls) == 1
    _, update = collection.update_many_calls[0]
    assert update["$set"]["metadata.amendment_status"] == "superseded"


def test_repeated_transitions_each_propagate_the_latest_state() -> None:
    """verify -> reject -> supersede, each one leaving the chunks agreeing
    with whatever the source's CURRENT state is, not an earlier one."""
    service = _service()
    source, collection = _linked_source_with_recording_embeddings(service)
    collection.update_many_calls.clear()  # only care about propagation from the transitions below, not the link
    new = _created(service, title="Replacement Act")

    import app.repositories.documents as documents_module

    embeddings = _RecordingEmbeddings()
    embeddings.collection = collection
    original = documents_module.EmbeddingMetadataRepository
    documents_module.EmbeddingMetadataRepository = lambda: embeddings  # type: ignore[misc,assignment]
    try:
        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="verified", last_verified_date=date(2026, 1, 10),
            evidence_url="https://www.indiacode.nic.in/x", review_notes="Checked.",
        )))
        assert collection.update_many_calls[-1][1]["$set"]["metadata.verification_status"] == "verified"

        asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
            verification_status="rejected", last_verified_date=date(2026, 1, 20),
            review_notes="Later found to be an unofficial reprint.",
        )))
        assert collection.update_many_calls[-1][1]["$set"]["metadata.verification_status"] == "rejected"

        asyncio.run(service.supersede(source.source_id, ADMIN, SupersedeSourceRequest(
            superseded_by_source_id=new.source_id, review_notes="Formally replaced.",
        )))
        assert collection.update_many_calls[-1][1]["$set"]["metadata.amendment_status"] == "superseded"
    finally:
        documents_module.EmbeddingMetadataRepository = original  # type: ignore[misc]

    assert len(collection.update_many_calls) == 3


def test_a_source_never_linked_to_a_document_propagates_nothing() -> None:
    """No `document_id` yet -- propagation must be a no-op, not an error."""
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 15),
        evidence_url="https://www.indiacode.nic.in/x",
        review_notes="Checked.",
    )))
    assert result.verification_status == "verified"


# ---------------------------------------------------------------------------
# G2 -- POST /verify must actually be able to reach `verified`
# ---------------------------------------------------------------------------


def test_verify_endpoint_request_can_actually_reach_verified() -> None:
    """Before this fix, `VerifySourceRequest` had no `evidence_url`/
    `review_notes` fields at all, so `verify()`'s delegation to `review()`
    always sent empty strings for both regardless of what a caller supplied
    -- `POST /verify` could never satisfy `review()`'s own requirement that a
    `verified` transition carry real evidence, so it always raised
    `BadRequestError`, on every call, for every source."""
    service = _service()
    source = _created(service)

    result = asyncio.run(service.verify(source.source_id, ADMIN, VerifySourceRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 15),
        evidence_url="https://www.indiacode.nic.in/handle/123456789/2000",
        review_notes="Compared against the gazette text.",
    )))

    assert result.verification_status == "verified"
    assert result.evidence_url == "https://www.indiacode.nic.in/handle/123456789/2000"
    assert result.review_notes == "Compared against the gazette text."


def test_verify_endpoint_still_rejects_a_verified_claim_with_no_evidence_anywhere() -> None:
    """G2's fix must not weaken the underlying evidence requirement -- a
    verify call with no evidence, and no prior evidence already on file,
    must still be refused."""
    service = _service()
    source = _created(service)
    with pytest.raises(BadRequestError):
        asyncio.run(service.verify(source.source_id, ADMIN, VerifySourceRequest(
            verification_status="verified",
            last_verified_date=date(2026, 1, 15),
        )))


# ---------------------------------------------------------------------------
# G3 -- re-review must not erase evidence, and staleness must reflect status
# ---------------------------------------------------------------------------


def test_omitting_evidence_on_a_later_review_does_not_erase_it() -> None:
    """A source verified once, then later moved to `rejected` without
    re-typing the same evidence URL, must keep that evidence on file --
    omitted means 'unchanged', not 'clear it'."""
    service = _service()
    source = _created(service)
    asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 10),
        evidence_url="https://www.indiacode.nic.in/original-evidence",
        review_notes="Initial verification.",
    )))

    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="rejected",
        last_verified_date=date(2026, 1, 20),
        review_notes="Later found to be an unofficial reprint.",
    )))

    assert result.verification_status == "rejected"
    assert result.evidence_url == "https://www.indiacode.nic.in/original-evidence"


def test_explicitly_clearing_evidence_with_an_empty_string_is_still_honoured() -> None:
    """Omitting must preserve, but an admin who explicitly sends an empty
    string is making a real choice to clear the field, and that must still
    work -- G3 is about accidental erasure, not making the field immutable."""
    service = _service()
    source = _created(service)
    asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date(2026, 1, 10),
        evidence_url="https://www.indiacode.nic.in/original-evidence",
        review_notes="Initial verification.",
    )))

    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="rejected",
        last_verified_date=date(2026, 1, 20),
        evidence_url="",
        review_notes="Evidence link found to be a dead link; cleared pending re-check.",
    )))

    assert result.evidence_url == ""


def test_a_rejected_source_is_stale_even_with_a_recent_review_date() -> None:
    """Security finding G3: staleness must reflect verification status, not
    only age -- a source reviewed TODAY and rejected is not "fresh"."""
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="rejected",
        last_verified_date=date.today(),
        review_notes="Unofficial reprint.",
    )))
    assert result.stale is True


def test_a_pending_review_source_is_stale_even_with_a_recent_date() -> None:
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="pending_review",
        last_verified_date=date.today(),
    )))
    assert result.stale is True


def test_a_freshly_verified_source_is_not_stale() -> None:
    service = _service()
    source = _created(service)
    result = asyncio.run(service.review(source.source_id, ADMIN, SourceReviewRequest(
        verification_status="verified",
        last_verified_date=date.today(),
        evidence_url="https://www.indiacode.nic.in/x",
        review_notes="Checked today.",
    )))
    assert result.stale is False
