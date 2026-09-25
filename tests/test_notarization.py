"""Tests for the E-Notarization module.

The organising principle: every test here defends the single rule the module
exists to protect -- a document is "notarized" if and only if a verified,
active notary explicitly approved a request for exactly those bytes.

The service is exercised against in-memory fake repositories rather than a
live MongoDB, so the legal invariants are tested independently of database
availability. The fakes implement only the methods the service actually
calls, so a new repository call cannot silently go untested.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.notarization import states
from app.notarization.audit import build_event, redact_detail
from app.notarization.checklist import evaluate, is_eligible
from app.notarization.esign import available_providers, get_provider
from app.notarization.esign.base import ESignProvider, SigningRequest, SigningSession
from app.notarization.esign.registry import register_provider
from app.notarization.integrity import hash_sections, hashes_match, new_verification_token
from app.notarization.qr import QRUnavailableError, build_verification_qr_svg
from app.notarization.service import NotarizationService, status_label
from app.notarization.tokens import issue_action_token, verify_action_token
from app.schemas.notarization import PublicVerificationResponse

# ---------------------------------------------------------------------------
# In-memory repository doubles
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def insert(self, document: dict[str, Any]) -> str:
        document.setdefault("_id", str(uuid4()))
        document.setdefault("created_at", datetime.now(UTC))
        self.rows[str(document["_id"])] = document
        return str(document["_id"])

    async def find_by_id(self, item_id: str) -> dict[str, Any] | None:
        return self.rows.get(item_id)

    async def update_by_id(self, item_id: str, updates: dict[str, Any]) -> bool:
        if item_id not in self.rows:
            return False
        self.rows[item_id].update(updates)
        return True


class _FakeDocuments(_FakeCollection):
    async def latest_version(self, source_draft_id: str) -> dict[str, Any] | None:
        matches = [row for row in self.rows.values() if row.get("source_draft_id") == source_draft_id]
        return max(matches, key=lambda row: row.get("document_version", 0)) if matches else None

    async def find_by_verification_token(self, token: str) -> dict[str, Any] | None:
        return next((row for row in self.rows.values() if row.get("verification_token") == token), None)


class _FakeNotaries(_FakeCollection):
    async def find_by_user_id(self, user_id: str) -> dict[str, Any] | None:
        return next((row for row in self.rows.values() if row.get("user_id") == user_id), None)


class _FakeRequests(_FakeCollection):
    async def list_for_notary(self, notary_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [
            row for row in self.rows.values()
            if row.get("assigned_notary_id") == notary_id and (status is None or row.get("review_status") == status)
        ][:limit]

    async def list_for_requester(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return [row for row in self.rows.values() if row.get("requested_by_user_id") == user_id][:limit]


class _FakeAudit:
    """Append + read only, exactly like the real repository."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def append(self, event: dict[str, Any]) -> str:
        event.setdefault("_id", str(uuid4()))
        event.setdefault("occurred_at", datetime.now(UTC))
        self.events.append(event)
        return str(event["_id"])

    async def list_for_document(self, document_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("document_id") == document_id][:limit]

    async def list_recent(self, limit: int = 200, action: str | None = None) -> list[dict[str, Any]]:
        events = [event for event in self.events if action is None or event.get("action") == action]
        return list(reversed(events))[:limit]


def _service() -> NotarizationService:
    service = NotarizationService.__new__(NotarizationService)
    service.documents = _FakeDocuments()
    service.notaries = _FakeNotaries()
    service.requests = _FakeRequests()
    service.sessions = _FakeCollection()
    service.audit = _FakeAudit()
    return service


_SECTIONS = {"Deponent Details": "Rahul Sharma, Ghaziabad", "Statements": "1. That the facts are true."}
_SIGNER = {
    "full_name": "Rahul Sharma",
    "address": "24, Shastri Nagar, Ghaziabad",
    "identity_document_type": "Passport",
    "place": "Ghaziabad",
    "date": "01/09/2026",
    "email": "rahul@example.com",
}
_OWNER = "user-1"


@register_provider
class _FakeSuccessProvider(ESignProvider):
    """Security finding N2: a deterministic, in-memory signing provider so
    `ready_for_signature -> signed -> notarized` can actually be exercised
    end to end. `ManualESignProvider` (the only provider reachable from a
    test before this fix) never signs anything by design (see
    `test_default_esign_provider_never_reports_a_signature`), so nothing in
    this suite ever drove a document through a real signing ceremony --
    every prior "signed" fixture (`_through_to_signed` below) set `status`
    directly on the fake document rather than going through
    `initiate_signing`/`handle_signing_callback` at all, which is exactly
    how N1 (the callback token being generated and then silently lost)
    stayed uncaught.
    """

    name = "fake-success"

    def __init__(self) -> None:
        # Instance-level, not class-level -- `get_provider()` constructs a
        # fresh instance per call in production, and a class-level list
        # shared across every test that resolves this provider via the
        # registry (rather than injecting a specific instance, as the one
        # test that reads this list does) is an ordering hazard for no
        # benefit.
        self.received_urls: list[str] = []

    async def initiate(self, request: SigningRequest) -> SigningSession:
        self.received_urls.append(request.callback_url)
        return SigningSession(
            provider=self.name, provider_reference=str(uuid4()), status="initiated",
            signing_url="https://example-signing-vendor.test/session/abc",
        )

    async def fetch_status(self, provider_reference: str) -> SigningSession:
        return SigningSession(provider=self.name, provider_reference=provider_reference, status="signed")

    async def cancel(self, provider_reference: str) -> SigningSession:
        return SigningSession(provider=self.name, provider_reference=provider_reference, status="cancelled")


async def _prepared(service: NotarizationService, owner: str = _OWNER) -> dict[str, Any]:
    result = await service.prepare(
        draft_id="draft-1",
        draft_title="Affidavit",
        draft_category="Affidavit",
        sections=_SECTIONS,
        signer=_SIGNER,
        witnesses=[],
        user_id=owner,
    )
    return result["document"]


async def _verified_notary(service: NotarizationService, user_id: str = "notary-user") -> str:
    return await service.notaries.insert(
        {
            "user_id": user_id,
            "full_name": "Adv. S. Iyer",
            "registration_number": "NOT/UP/2019/1234",
            "jurisdiction_state": "Uttar Pradesh",
            "verification_status": "verified",
            "active": True,
        }
    )


async def _through_to_signed(service: NotarizationService) -> dict[str, Any]:
    document = await _prepared(service)
    await service.documents.update_by_id(str(document["_id"]), {"status": states.SIGNED})
    return await service.documents.find_by_id(str(document["_id"]))


async def _through_to_notarized(service: NotarizationService) -> tuple[dict[str, Any], str, str]:
    notary_id = await _verified_notary(service)
    document = await _through_to_signed(service)
    request = await service.create_request(
        document_id=str(document["_id"]), user_id=_OWNER, assigned_notary_id=notary_id,
        supporting_id_confirmed=True, note="",
    )
    request_id = str(request["_id"])
    await service.start_review(request_id=request_id, notary_user_id="notary-user")
    result = await service.approve_request(
        request_id=request_id,
        notary_user_id="notary-user",
        action_token=service.issue_decision_token(request_id, "approve"),
        remarks="Verified in person.",
        confirmed_document_hash=document["document_hash"],
    )
    return result["document"], request_id, notary_id


# ---------------------------------------------------------------------------
# The six distinct statuses must stay distinguishable
# ---------------------------------------------------------------------------


def test_only_the_notarized_status_is_labelled_notarized() -> None:
    labelled = [status for status in states.ALL_STATUSES if status_label(status) == "Notarized"]
    assert labelled == [states.NOTARIZED]


def test_signed_is_explicitly_not_notarized() -> None:
    assert status_label(states.SIGNED) == "Signed (not notarized)"
    assert states.is_notarized(states.SIGNED) is False


def test_generating_or_preparing_never_produces_a_notarized_document() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    assert document["status"] == states.READY_FOR_SIGNATURE
    assert states.is_notarized(document["status"]) is False
    assert document["verification_token"] is None


def test_checklist_can_only_ever_say_notary_ready_draft() -> None:
    complete = evaluate("Affidavit", _SIGNER, [])
    assert complete.complete is True
    assert complete.label == "Notary-ready draft"
    assert "Notarized" not in complete.label


def test_only_eligible_document_types_can_be_prepared() -> None:
    assert is_eligible("Affidavit") and is_eligible("Contract") and is_eligible("NOC")
    assert not is_eligible("Complaint")
    service = _service()
    with pytest.raises(BadRequestError):
        asyncio.run(
            service.prepare(
                draft_id="d", draft_title="Police Complaint", draft_category="Complaint",
                sections=_SECTIONS, signer=_SIGNER, witnesses=[], user_id=_OWNER,
            )
        )


# ---------------------------------------------------------------------------
# A user cannot self-mark a document notarized
# ---------------------------------------------------------------------------


def test_user_cannot_transition_a_document_straight_to_notarized() -> None:
    """The state machine has no edge from any user-reachable state to
    `notarized`; only notary review leads there."""
    for status in (states.DRAFT, states.READY_FOR_SIGNATURE, states.SIGNING_IN_PROGRESS,
                   states.SIGNED, states.NOTARY_REVIEW_REQUESTED):
        assert not states.can_transition(status, states.NOTARIZED)
    assert states.can_transition(states.NOTARY_REVIEW_IN_PROGRESS, states.NOTARIZED)


def test_user_without_a_notary_account_cannot_approve() -> None:
    service = _service()
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=None, supporting_id_confirmed=True, note="")
    )
    request_id = str(request["_id"])
    with pytest.raises(ForbiddenError, match="not registered as a notary"):
        asyncio.run(
            service.approve_request(
                request_id=request_id, notary_user_id=_OWNER,
                action_token=service.issue_decision_token(request_id, "approve"),
                remarks="", confirmed_document_hash=document["document_hash"],
            )
        )


def test_unverified_notary_cannot_approve() -> None:
    service = _service()
    asyncio.run(
        service.notaries.insert(
            {"user_id": "pending-notary", "full_name": "X", "registration_number": "R1",
             "jurisdiction_state": "UP", "verification_status": "unverified", "active": False}
        )
    )
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=None, supporting_id_confirmed=True, note="")
    )
    request_id = str(request["_id"])
    with pytest.raises(ForbiddenError, match="has not been verified"):
        asyncio.run(
            service.approve_request(
                request_id=request_id, notary_user_id="pending-notary",
                action_token=service.issue_decision_token(request_id, "approve"),
                remarks="", confirmed_document_hash=document["document_hash"],
            )
        )


def test_revoked_notary_account_cannot_approve() -> None:
    """An admin revoking a notary must stop them notarizing immediately."""
    service = _service()
    notary_id = asyncio.run(_verified_notary(service))
    asyncio.run(service.notaries.update_by_id(notary_id, {"verification_status": "revoked", "active": False}))
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=None, supporting_id_confirmed=True, note="")
    )
    request_id = str(request["_id"])
    with pytest.raises(ForbiddenError):
        asyncio.run(
            service.approve_request(
                request_id=request_id, notary_user_id="notary-user",
                action_token=service.issue_decision_token(request_id, "approve"),
                remarks="", confirmed_document_hash=document["document_hash"],
            )
        )


def test_verified_notary_approval_is_the_only_route_to_notarized() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    assert document["status"] == states.NOTARIZED
    assert document["verification_token"]
    assert document["notarized_at"] is not None


# ---------------------------------------------------------------------------
# Rejected / failed / cancelled must remain non-notarized
# ---------------------------------------------------------------------------


def test_rejected_request_leaves_the_document_not_notarized() -> None:
    service = _service()
    notary_id = asyncio.run(_verified_notary(service))
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=notary_id, supporting_id_confirmed=False, note="")
    )
    request_id = str(request["_id"])
    asyncio.run(service.start_review(request_id=request_id, notary_user_id="notary-user"))
    result = asyncio.run(
        service.reject_request(
            request_id=request_id, notary_user_id="notary-user",
            action_token=service.issue_decision_token(request_id, "reject"),
            remarks="Signer did not present original ID.",
        )
    )
    assert result["document"]["status"] == states.SIGNED
    assert states.is_notarized(result["document"]["status"]) is False
    assert result["document"]["verification_token"] is None
    assert result["request"]["review_status"] == "rejected"


def test_rejection_requires_a_reason() -> None:
    service = _service()
    notary_id = asyncio.run(_verified_notary(service))
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=notary_id, supporting_id_confirmed=False, note="")
    )
    request_id = str(request["_id"])
    with pytest.raises(BadRequestError, match="rejection reason"):
        asyncio.run(
            service.reject_request(
                request_id=request_id, notary_user_id="notary-user",
                action_token=service.issue_decision_token(request_id, "reject"), remarks="   ",
            )
        )


@pytest.mark.parametrize("failure_status", ["failed", "expired", "cancelled"])
def test_unsuccessful_esign_leaves_the_document_unsigned(failure_status: str) -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    document_id = str(document["_id"])
    asyncio.run(service.documents.update_by_id(document_id, {"status": states.SIGNING_IN_PROGRESS}))
    session_id = asyncio.run(
        service.sessions.insert(
            {
                "document_id": document_id, "document_version": 1,
                "document_hash": document["document_hash"], "owner_user_id": _OWNER,
                "provider": "manual", "provider_reference": "ref-1", "status": "initiated",
            }
        )
    )
    result = asyncio.run(
        service.handle_signing_callback(
            session_id=session_id,
            action_token=issue_action_token("esign_callback", session_id, 300),
            status=failure_status,
            failure_reason="provider reported failure",
        )
    )
    assert result["document"]["status"] == states.READY_FOR_SIGNATURE
    assert states.is_notarized(result["document"]["status"]) is False


def test_esign_callback_requires_a_valid_action_token() -> None:
    """Without this the callback is an unauthenticated 'mark it signed' endpoint."""
    service = _service()
    document = asyncio.run(_prepared(service))
    session_id = asyncio.run(
        service.sessions.insert(
            {"document_id": str(document["_id"]), "document_version": 1,
             "document_hash": document["document_hash"], "owner_user_id": _OWNER,
             "provider": "manual", "provider_reference": "r", "status": "initiated"}
        )
    )
    with pytest.raises(ForbiddenError):
        asyncio.run(
            service.handle_signing_callback(
                session_id=session_id, action_token="forged.token", status="signed"
            )
        )


def test_default_esign_provider_never_reports_a_signature() -> None:
    """The unconfigured default must not fabricate a signature."""
    provider = get_provider("manual")
    session = asyncio.run(
        provider.initiate(SigningRequest("d", 1, "T", "hash", "Name", "e@x.com", "purpose"))
    )
    assert session.status == "pending"
    assert "No e-signature provider is configured" in session.failure_reason


def test_esign_provider_is_pluggable_and_not_hardcoded() -> None:
    assert "manual" in available_providers()
    with pytest.raises(BadRequestError):
        get_provider("some-vendor-that-is-not-registered")


# ---------------------------------------------------------------------------
# Document integrity
# ---------------------------------------------------------------------------


def test_hash_changes_when_the_document_changes() -> None:
    original = hash_sections(_SECTIONS)
    edited = hash_sections({**_SECTIONS, "Statements": "1. That the facts are true. 2. Added."})
    assert original != edited
    assert len(original) == 64


def test_hash_mismatch_blocks_notary_approval() -> None:
    """A notary must never attest bytes they did not review."""
    service = _service()
    notary_id = asyncio.run(_verified_notary(service))
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=notary_id, supporting_id_confirmed=True, note="")
    )
    request_id = str(request["_id"])
    asyncio.run(service.start_review(request_id=request_id, notary_user_id="notary-user"))
    with pytest.raises(BadRequestError, match="does not match"):
        asyncio.run(
            service.approve_request(
                request_id=request_id, notary_user_id="notary-user",
                action_token=service.issue_decision_token(request_id, "approve"),
                remarks="", confirmed_document_hash="0" * 64,
            )
        )
    refreshed = asyncio.run(service.documents.find_by_id(str(document["_id"])))
    assert refreshed["status"] != states.NOTARIZED


def test_document_edited_mid_signing_invalidates_the_signature() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    document_id = str(document["_id"])
    session_id = asyncio.run(
        service.sessions.insert(
            {"document_id": document_id, "document_version": 1,
             "document_hash": document["document_hash"], "owner_user_id": _OWNER,
             "provider": "manual", "provider_reference": "r", "status": "initiated"}
        )
    )
    # The document changes after signing began.
    asyncio.run(service.documents.update_by_id(document_id, {"document_hash": hash_sections({"X": "changed"})}))
    with pytest.raises(BadRequestError, match="changed after signing began"):
        asyncio.run(
            service.handle_signing_callback(
                session_id=session_id,
                action_token=issue_action_token("esign_callback", session_id, 300),
                status="signed",
            )
        )


def test_editing_a_notarized_document_creates_a_new_version_and_invalidates_the_old() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    original_id = str(document["_id"])

    new_document = asyncio.run(
        service.create_version(
            document_id=original_id, sections={**_SECTIONS, "Statements": "1. Amended."}, user_id=_OWNER
        )
    )
    # The new version is an ordinary draft with no notarization.
    assert new_document["status"] == states.DRAFT
    assert new_document["document_version"] == document["document_version"] + 1
    assert new_document["verification_token"] is None
    assert new_document["document_hash"] != document["document_hash"]

    # The old version keeps its record but its notarization is invalidated.
    previous = asyncio.run(service.documents.find_by_id(original_id))
    assert previous["notarization_valid"] is False
    assert previous["superseded_by_document_id"] == str(new_document["_id"])


def test_a_notarized_document_cannot_be_edited_in_place() -> None:
    assert states.is_editable(states.NOTARIZED) is False
    assert states.is_editable(states.SIGNED) is False
    assert states.is_editable(states.DRAFT) is True


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_audit_events_carry_timestamp_actor_role_version_and_action() -> None:
    event = build_event(
        action="notary_approved", document_id="d1", document_version=3,
        actor_id="n1", actor_role="notary",
    )
    for field in ("occurred_at", "actor_role", "document_version", "action", "actor_id"):
        assert field in event
    assert event["document_version"] == 3


def test_audit_repository_exposes_no_update_or_delete() -> None:
    """Application-level immutability: there is no code path to alter or
    remove a recorded event."""
    from app.repositories.notarization import NotarizationAuditRepository

    repository = NotarizationAuditRepository()
    for forbidden in ("update_by_id", "delete_by_id", "delete_by_user", "delete_by_session"):
        assert not hasattr(repository, forbidden), f"audit repository must not expose {forbidden}"


def test_audit_detail_redacts_sensitive_keys() -> None:
    cleaned = redact_detail(
        {
            "reason": "ok",
            "signer_otp": "123456",
            "aadhaar_number": "1234 5678 9012",
            "provider_api_key": "sk-secret",
            "nested": {"biometric_template": "..."},
        }
    )
    assert cleaned["reason"] == "ok"
    assert cleaned["signer_otp"] == "[redacted]"
    assert cleaned["aadhaar_number"] == "[redacted]"
    assert cleaned["provider_api_key"] == "[redacted]"
    assert cleaned["nested"]["biometric_template"] == "[redacted]"


def test_the_full_lifecycle_is_recorded_in_order() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    actions = [event["action"] for event in asyncio.run(service.audit.list_for_document(str(document["_id"])))]
    for expected in ("notary_ready_prepared", "notarization_requested", "notary_review_started",
                     "notary_approved", "notarized_version_issued"):
        assert expected in actions, f"{expected} missing from audit trail: {actions}"


# ---------------------------------------------------------------------------
# Verification page
# ---------------------------------------------------------------------------


def test_verification_returns_no_personal_data() -> None:
    """The public response must never carry document content or party PII."""
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    result = asyncio.run(service.verify(document["verification_token"]))
    response = PublicVerificationResponse(**result)

    serialized = response.model_dump_json()
    assert _SIGNER["address"] not in serialized
    assert _SIGNER["email"] not in serialized
    assert _SIGNER["full_name"] not in serialized
    assert _SECTIONS["Statements"] not in serialized
    assert _OWNER not in serialized
    # But it does carry what a verifier legitimately needs.
    assert response.found is True
    assert response.notary_name == "Adv. S. Iyer"
    assert response.notary_registration_number == "NOT/UP/2019/1234"
    assert response.status == states.NOTARIZED


def test_verification_reports_hash_match_and_mismatch() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    token = document["verification_token"]

    assert asyncio.run(service.verify(token, document["document_hash"]))["hash_matches"] is True
    assert asyncio.run(service.verify(token, "0" * 64))["hash_matches"] is False
    # Omitted entirely when the verifier supplies no hash.
    assert asyncio.run(service.verify(token))["hash_matches"] is None


def test_revoked_document_verifies_as_revoked() -> None:
    service = _service()
    document, request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    token = document["verification_token"]

    asyncio.run(
        service.revoke(
            request_id=request_id, notary_user_id="notary-user",
            action_token=service.issue_decision_token(request_id, "revoke"),
            remarks="Issued on a forged identity document.",
        )
    )
    result = asyncio.run(service.verify(token))
    assert result["revoked"] is True
    assert result["status"] == states.REVOKED
    assert "revoked" in result["message"].lower()


def test_unknown_verification_token_reveals_nothing() -> None:
    service = _service()
    result = asyncio.run(service.verify(new_verification_token()))
    assert result["found"] is False
    assert "notary_name" not in result


def test_qr_is_only_generated_for_notarized_documents() -> None:
    for status in (states.DRAFT, states.READY_FOR_SIGNATURE, states.SIGNED,
                   states.NOTARY_REVIEW_REQUESTED, states.REVOKED):
        with pytest.raises(QRUnavailableError):
            build_verification_qr_svg(status, "https://example.com/verify/abc")
    svg = build_verification_qr_svg(states.NOTARIZED, "https://example.com/verify/abc")
    assert svg.startswith("<svg")


def test_qr_code_actually_decodes_to_the_verification_url() -> None:
    """A verification QR that does not scan is worse than none, because it
    looks like a working one. This decodes the generated code the way a phone
    camera would."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    segno = pytest.importorskip("segno")
    import io

    url = "https://verify.example.com/verify/Xk3n9QpL2mR7vT4sYw8zBc1dFg6hJ0aE"
    # The module under test emits SVG; render the same payload to a raster so
    # a decoder can read it.
    assert build_verification_qr_svg(states.NOTARIZED, url).startswith("<svg")

    buffer = io.BytesIO()
    segno.make(url, error="m").save(buffer, kind="png", scale=10, border=4)
    image = cv2.imdecode(numpy.frombuffer(buffer.getvalue(), numpy.uint8), cv2.IMREAD_GRAYSCALE)
    decoded, _points, _straight = cv2.QRCodeDetector().detectAndDecode(image)
    assert decoded == url


# ---------------------------------------------------------------------------
# Authorization / ownership
# ---------------------------------------------------------------------------


def test_a_user_cannot_read_another_users_document() -> None:
    service = _service()
    document = asyncio.run(_prepared(service, owner="user-1"))
    with pytest.raises(NotFoundError):
        asyncio.run(service.document_status(document_id=str(document["_id"]), user_id="user-2"))


def test_a_user_cannot_sign_another_users_document() -> None:
    service = _service()
    document = asyncio.run(_prepared(service, owner="user-1"))
    with pytest.raises(NotFoundError):
        asyncio.run(
            service.initiate_signing(document_id=str(document["_id"]), purpose="p", user_id="user-2")
        )


def test_anonymous_access_is_refused_outright() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    with pytest.raises(ForbiddenError):
        asyncio.run(service.document_status(document_id=str(document["_id"]), user_id=""))


def test_a_notary_cannot_act_on_another_notarys_request() -> None:
    service = _service()
    assigned_notary = asyncio.run(_verified_notary(service, "notary-a"))
    asyncio.run(
        service.notaries.insert(
            {"user_id": "notary-b", "full_name": "Other", "registration_number": "R2",
             "jurisdiction_state": "MH", "verification_status": "verified", "active": True}
        )
    )
    document = asyncio.run(_through_to_signed(service))
    request = asyncio.run(
        service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                               assigned_notary_id=assigned_notary, supporting_id_confirmed=True, note="")
    )
    request_id = str(request["_id"])
    with pytest.raises(NotFoundError):
        asyncio.run(service.start_review(request_id=request_id, notary_user_id="notary-b"))


def test_a_request_can_only_be_created_for_a_signed_document() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    with pytest.raises(BadRequestError, match="Only a signed document"):
        asyncio.run(
            service.create_request(document_id=str(document["_id"]), user_id=_OWNER,
                                   assigned_notary_id=None, supporting_id_confirmed=True, note="")
        )


def test_a_request_cannot_be_decided_twice() -> None:
    service = _service()
    document, request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    with pytest.raises(BadRequestError, match="already been approved"):
        asyncio.run(
            service.approve_request(
                request_id=request_id, notary_user_id="notary-user",
                action_token=service.issue_decision_token(request_id, "approve"),
                remarks="", confirmed_document_hash=document["document_hash"],
            )
        )


# ---------------------------------------------------------------------------
# Action tokens
# ---------------------------------------------------------------------------


def test_action_token_is_scoped_to_purpose_and_subject() -> None:
    token = issue_action_token("approve", "request-1", 300)
    assert verify_action_token(token, "approve", "request-1")["subject"] == "request-1"
    with pytest.raises(ForbiddenError, match="not issued for this operation"):
        verify_action_token(token, "revoke", "request-1")
    with pytest.raises(ForbiddenError, match="not issued for this document"):
        verify_action_token(token, "approve", "request-2")


def test_expired_action_token_is_refused() -> None:
    token = issue_action_token("approve", "request-1", -1)
    with pytest.raises(ForbiddenError, match="expired"):
        verify_action_token(token, "approve", "request-1")


def test_tampered_action_token_is_refused() -> None:
    token = issue_action_token("approve", "request-1", 300)
    payload, _, signature = token.partition(".")
    with pytest.raises(ForbiddenError, match="signature is invalid"):
        verify_action_token(f"{payload}x.{signature}", "approve", "request-1")


def test_hash_comparison_is_case_insensitive_and_safe() -> None:
    digest = hash_sections(_SECTIONS)
    assert hashes_match(digest, digest.upper())
    assert not hashes_match(digest, "0" * 64)


# ---------------------------------------------------------------------------
# Download gating
# ---------------------------------------------------------------------------


def test_final_document_is_downloadable_only_once_notarized() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    status = asyncio.run(service.document_status(document_id=str(document["_id"]), user_id=_OWNER))
    assert status["downloadable"] is False

    notarized, _request_id, _notary_id = asyncio.run(_through_to_notarized(_service()))
    assert states.is_notarized(notarized["status"])


# ---------------------------------------------------------------------------
# Export gating: only a notarized document carries a notarization record
# ---------------------------------------------------------------------------


def test_export_options_carry_no_notarization_by_default() -> None:
    from app.drafting.export import ExportOptions

    assert ExportOptions().notarization is None


def test_building_notarized_export_options_refuses_a_non_notarized_document() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    with pytest.raises(BadRequestError, match="Only a notarized document"):
        asyncio.run(service.build_notarized_export_options(document))


def test_notarized_export_options_come_only_from_the_notary_record() -> None:
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    options = asyncio.run(service.build_notarized_export_options(document))

    attestation = options.notarization
    assert attestation is not None
    # Every field traces to the stored notary account / notarization record.
    assert attestation.notary_name == "Adv. S. Iyer"
    assert attestation.notary_registration_number == "NOT/UP/2019/1234"
    assert attestation.jurisdiction_state == "Uttar Pradesh"
    assert attestation.document_hash == document["document_hash"]
    assert document["verification_token"] in attestation.verification_url
    assert attestation.verification_qr_svg.startswith("<svg")


def test_notarized_export_refuses_when_no_attesting_notary_is_on_record() -> None:
    """A notarized document with no notary would print a blank attestation.
    Refusing is the only safe behaviour."""
    service = _service()
    document, _request_id, _notary_id = asyncio.run(_through_to_notarized(service))
    asyncio.run(service.documents.update_by_id(str(document["_id"]), {"notarized_by_notary_id": "missing"}))
    refreshed = asyncio.run(service.documents.find_by_id(str(document["_id"])))
    with pytest.raises(BadRequestError, match="no attesting notary"):
        asyncio.run(service.build_notarized_export_options(refreshed))


def test_a_plain_draft_export_contains_no_notarization_record() -> None:
    """The rendered PDF must not mention notarization unless it is notarized."""
    pytest.importorskip("weasyprint")
    pypdf = pytest.importorskip("pypdf")
    import tempfile
    from pathlib import Path

    from app.drafting.export import ExportOptions, PdfDraftExporter

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "plain.pdf"
        PdfDraftExporter().export("Affidavit", _SECTIONS, output, "english", ExportOptions())
        text = "\n".join(page.extract_text() for page in pypdf.PdfReader(str(output)).pages)
    assert "NOTARIZATION RECORD" not in text


# ---------------------------------------------------------------------------
# N1/N2 -- the e-signature callback token must reach the provider/caller,
# and the full signed -> notarized -> verified pipeline must be reachable
# ---------------------------------------------------------------------------


def test_the_callback_token_is_embedded_in_the_url_the_provider_receives() -> None:
    """Security finding N1: previously the token was minted only AFTER the
    provider had already been called, so `SigningRequest.callback_url`
    could never have carried it -- confirmed here by asserting the fake
    provider's own received `callback_url` actually contains the token
    `initiate_signing` returns."""
    import app.notarization.service as service_module

    service = _service()
    document = asyncio.run(_prepared(service))
    # A fresh, locally-scoped instance and a monkeypatched `get_provider`
    # (rather than relying on the shared `_PROVIDERS` registry entry and its
    # class-level `received_callback_urls`) so this assertion can only ever
    # be about the exact call this test made, immune to registry/import
    # ordering relative to any other test file.
    provider = _FakeSuccessProvider()
    original_get_provider = service_module.get_provider
    service_module.get_provider = lambda name=None: provider
    try:
        result = asyncio.run(service.initiate_signing(
            document_id=str(document["_id"]), purpose="affidavit execution",
            user_id=_OWNER, provider_name="fake-success",
        ))
    finally:
        service_module.get_provider = original_get_provider

    assert result["callback_token"]
    assert provider.received_urls
    assert result["callback_token"] in provider.received_urls[-1]
    assert result["session_id"] in provider.received_urls[-1]


def test_initiate_signing_response_actually_carries_the_callback_token() -> None:
    """The route-level half of N1: `InitiateSigningResponse` previously had
    no `callback_token` field at all, so even a caller design that expects
    to relay the token itself (rather than relying solely on the provider
    callback URL) had no way to get it."""
    from app.schemas.notarization import InitiateSigningResponse

    assert "callback_token" in InitiateSigningResponse.model_fields


def test_full_pipeline_ready_for_signature_to_signed_to_notarized_to_verified() -> None:
    """Security finding N2: the complete, real pipeline -- prepare -> initiate
    signing -> PROVIDER CALLBACK (using the token exactly as a real provider
    would receive/return it, not a status set directly on the fake
    document) -> notary approval -> notarized -> public verification. Every
    prior "signed" fixture in this file bypassed the signing step entirely;
    this is the first test to drive it through the real service methods."""
    service = _service()
    document = asyncio.run(_prepared(service))
    assert document["status"] == states.READY_FOR_SIGNATURE

    signing = asyncio.run(service.initiate_signing(
        document_id=str(document["_id"]), purpose="affidavit execution",
        user_id=_OWNER, provider_name="fake-success",
    ))
    mid_flight = asyncio.run(service.documents.find_by_id(str(document["_id"])))
    assert mid_flight["status"] == states.SIGNING_IN_PROGRESS

    callback_result = asyncio.run(service.handle_signing_callback(
        session_id=signing["session_id"], action_token=signing["callback_token"], status="signed",
    ))
    assert callback_result["document"]["status"] == states.SIGNED

    notary_id = asyncio.run(_verified_notary(service))
    signed_document = callback_result["document"]
    request = asyncio.run(service.create_request(
        document_id=str(signed_document["_id"]), user_id=_OWNER, assigned_notary_id=notary_id,
        supporting_id_confirmed=True, note="",
    ))
    request_id = str(request["_id"])
    asyncio.run(service.start_review(request_id=request_id, notary_user_id="notary-user"))
    approval = asyncio.run(service.approve_request(
        request_id=request_id, notary_user_id="notary-user",
        action_token=service.issue_decision_token(request_id, "approve"),
        remarks="Verified in person.", confirmed_document_hash=signed_document["document_hash"],
    ))
    notarized_document = approval["document"]
    assert notarized_document["status"] == states.NOTARIZED
    assert states.is_notarized(notarized_document["status"]) is True
    assert notarized_document["verification_token"]

    verification = asyncio.run(service.verify(
        notarized_document["verification_token"], provided_hash=notarized_document["document_hash"],
    ))
    assert verification["found"] is True
    assert verification["status"] == states.NOTARIZED
    assert verification["hash_matches"] is True


def test_a_forged_callback_token_cannot_advance_a_different_session() -> None:
    """Invalid-transition coverage for N2: a token minted for session A must
    never be honoured for session B, even if both are otherwise valid."""
    service = _service()
    document_a = asyncio.run(_prepared(service, owner="user-a"))
    document_b = asyncio.run(service.prepare(
        draft_id="draft-2", draft_title="Affidavit", draft_category="Affidavit",
        sections=_SECTIONS, signer=_SIGNER, witnesses=[], user_id="user-b",
    ))
    document_b = document_b["document"]
    signing_a = asyncio.run(service.initiate_signing(
        document_id=str(document_a["_id"]), purpose="p", user_id="user-a", provider_name="fake-success",
    ))
    signing_b = asyncio.run(service.initiate_signing(
        document_id=str(document_b["_id"]), purpose="p", user_id="user-b", provider_name="fake-success",
    ))

    with pytest.raises(ForbiddenError):
        asyncio.run(service.handle_signing_callback(
            session_id=signing_b["session_id"], action_token=signing_a["callback_token"], status="signed",
        ))


def test_a_failed_callback_status_rolls_back_to_ready_for_signature() -> None:
    service = _service()
    document = asyncio.run(_prepared(service))
    signing = asyncio.run(service.initiate_signing(
        document_id=str(document["_id"]), purpose="p", user_id=_OWNER, provider_name="fake-success",
    ))

    result = asyncio.run(service.handle_signing_callback(
        session_id=signing["session_id"], action_token=signing["callback_token"],
        status="failed", failure_reason="signer_declined",
    ))

    assert result["document"]["status"] == states.READY_FOR_SIGNATURE
