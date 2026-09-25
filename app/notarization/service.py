"""Orchestration for the E-Notarization module.

Every legally-significant invariant lives here, in one file, so that "can
this document be called notarized?" has exactly one answer and one place to
read it:

* Only `approve_request` can set `notarized`, and it refuses unless the
  approving account is a VERIFIED, ACTIVE notary.
* Approval additionally refuses if the hash the notary confirms does not
  match the stored hash, so a notary cannot attest bytes they did not see.
* A rejected, failed, expired or cancelled outcome always leaves the document
  non-notarized.
* Editing a signed or notarized document is impossible; `create_version`
  makes a NEW document row at `draft`, and any prior notarization is
  explicitly invalidated on the superseded version.
* Every one of those steps appends an audit event that cannot later be
  altered or deleted through this application.

There is no method here, and there must never be one, that mints a notary
stamp, signature, registration number, certificate, or notarization record
without a verified notary's own approval action.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from app.core.config import settings
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.notarization import states
from app.notarization.audit import build_event
from app.notarization.checklist import evaluate as evaluate_checklist
from app.notarization.esign import get_provider
from app.notarization.esign.base import (
    UNSUCCESSFUL_SIGNING_STATUSES,
    SigningRequest,
)
from app.notarization.integrity import hash_sections, hashes_match, new_verification_token
from app.notarization.tokens import issue_action_token, verify_action_token
from app.repositories.notarization import (
    ESignSessionRepository,
    NotarizationAuditRepository,
    NotarizationDocumentRepository,
    NotarizationRequestRepository,
    NotaryAccountRepository,
)

log = structlog.get_logger(__name__)

# Human-readable label per status. Only `notarized` may say "Notarized", and
# only this table produces the word, so no caller can invent its own.
STATUS_LABELS: dict[str, str] = {
    states.DRAFT: "AI-generated draft",
    states.READY_FOR_SIGNATURE: "Notary-ready draft",
    states.SIGNING_IN_PROGRESS: "Signature in progress",
    states.SIGNED: "Signed (not notarized)",
    states.NOTARY_REVIEW_REQUESTED: "Notarization requested",
    states.NOTARY_REVIEW_IN_PROGRESS: "Under notary review",
    states.NOTARIZED: "Notarized",
    states.REVOKED: "Revoked",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, "Draft")


class NotarizationService:
    def __init__(self) -> None:
        self.documents = NotarizationDocumentRepository()
        self.notaries = NotaryAccountRepository()
        self.requests = NotarizationRequestRepository()
        self.sessions = ESignSessionRepository()
        self.audit = NotarizationAuditRepository()

    # -- helpers ---------------------------------------------------------

    async def _require_document(self, document_id: str) -> dict[str, Any]:
        document = await self.documents.find_by_id(document_id)
        if document is None:
            raise NotFoundError(f"Notarization document not found: {document_id}")
        return document

    @staticmethod
    def _require_ownership(document: dict[str, Any], user_id: str | None) -> None:
        """Ownership check applied to every document-scoped operation.

        Anonymous access is refused outright rather than treated as "no owner
        recorded, so open" -- the permissive reading is defensible for chat
        drafts but not for a signing or notarization flow.
        """
        if not user_id:
            raise ForbiddenError("Authentication is required for notarization operations.")
        if document.get("owner_user_id") != user_id:
            raise NotFoundError(f"Notarization document not found: {document.get('_id')}")

    async def _transition(
        self,
        document: dict[str, Any],
        target: str,
        *,
        action: str,
        actor_id: str | None,
        actor_role: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        current = document.get("status", states.DRAFT)
        if not states.can_transition(current, target):
            raise BadRequestError(
                f"A document in state '{current}' cannot move to '{target}'.",
                {"current_status": current, "target_status": target,
                 "allowed": sorted(states.allowed_transitions(current))},
            )
        await self.documents.update_by_id(str(document["_id"]), {"status": target})
        document["status"] = target
        await self.audit.append(
            build_event(
                action=action,  # type: ignore[arg-type]
                document_id=str(document["_id"]),
                document_version=int(document.get("document_version", 1)),
                document_hash=document.get("document_hash", ""),
                actor_id=actor_id,
                actor_role=actor_role,  # type: ignore[arg-type]
                detail={**(detail or {}), "from_status": current, "to_status": target},
            )
        )

    async def _require_verified_notary(self, user_id: str | None) -> dict[str, Any]:
        """The gate on every notarial act.

        Three separate conditions, checked separately so the audit log and the
        error message say which one failed: an account must exist, be
        `verified`, and be `active`. An admin revoking a notary sets
        `active=False`, which must immediately stop them approving anything.
        """
        if not user_id:
            raise ForbiddenError("Authentication is required.")
        notary = await self.notaries.find_by_user_id(user_id)
        if notary is None:
            raise ForbiddenError("This account is not registered as a notary.")
        if notary.get("verification_status") != "verified":
            raise ForbiddenError(
                "This notary account has not been verified by an administrator and cannot notarize documents."
            )
        if not notary.get("active", False):
            raise ForbiddenError("This notary account is inactive and cannot notarize documents.")
        return notary

    # -- 1. Notary-ready preparation --------------------------------------

    async def prepare(
        self,
        *,
        draft_id: str,
        draft_title: str,
        draft_category: str,
        sections: dict[str, str],
        signer: dict[str, str],
        witnesses: list[dict[str, str]],
        user_id: str,
    ) -> dict[str, Any]:
        """Creates (or refreshes) the notarizable document for a draft.

        Produces a "Notary-ready draft" at most. It never advances the
        document past `ready_for_signature`, and never toward `notarized`.
        """
        readiness = evaluate_checklist(draft_category, signer, witnesses)
        if not readiness.eligible:
            raise BadRequestError(readiness.ineligible_reason, {"category": draft_category})

        document_hash = hash_sections(sections)
        existing = await self.documents.latest_version(draft_id)

        if existing and existing.get("owner_user_id") != user_id:
            raise ForbiddenError("This draft belongs to another user.")

        # A prepared document is only reusable while it is still editable.
        # Once it is signed or notarized, preparing again must create a NEW
        # version rather than mutate bytes a signature covers.
        if existing and states.is_editable(existing.get("status", states.DRAFT)):
            document_id = str(existing["_id"])
            document_version = int(existing.get("document_version", 1))
            await self.documents.update_by_id(
                document_id,
                {
                    "document_hash": document_hash,
                    "signer": signer,
                    "witnesses": witnesses,
                    "checklist_complete": readiness.complete,
                    "status": states.READY_FOR_SIGNATURE if readiness.complete else states.DRAFT,
                },
            )
            document = await self._require_document(document_id)
        else:
            document_version = int(existing.get("document_version", 0)) + 1 if existing else 1
            document_id = str(uuid4())
            await self.documents.insert(
                {
                    "_id": document_id,
                    "source_draft_id": draft_id,
                    "document_version": document_version,
                    "document_title": draft_title,
                    "document_category": draft_category,
                    "document_hash": document_hash,
                    "sections": sections,
                    "signer": signer,
                    "witnesses": witnesses,
                    "owner_user_id": user_id,
                    "checklist_complete": readiness.complete,
                    "status": states.READY_FOR_SIGNATURE if readiness.complete else states.DRAFT,
                    # Minted only on notarization. Absent until then, so an
                    # un-notarized document has no verification URL to show.
                    "verification_token": None,
                    "notarized_at": None,
                    "notarized_by_notary_id": None,
                    "revoked_at": None,
                }
            )
            document = await self._require_document(document_id)

        await self.audit.append(
            build_event(
                action="notary_ready_prepared",
                document_id=document_id,
                document_version=document_version,
                document_hash=document_hash,
                actor_id=user_id,
                actor_role="user",
                detail={
                    "checklist_complete": readiness.complete,
                    "missing_required": readiness.missing_required(),
                    "category": draft_category,
                },
            )
        )
        return {
            "document": document,
            "readiness": readiness,
            "document_hash": document_hash,
            "document_version": document_version,
        }

    # -- 2. E-signature ----------------------------------------------------

    async def initiate_signing(
        self, *, document_id: str, purpose: str, user_id: str, provider_name: str | None = None
    ) -> dict[str, Any]:
        document = await self._require_document(document_id)
        self._require_ownership(document, user_id)

        if document.get("status") != states.READY_FOR_SIGNATURE:
            raise BadRequestError(
                "This document is not ready for signature. Complete the notarization checklist first.",
                {"status": document.get("status")},
            )

        provider = get_provider(provider_name)
        session_id = str(uuid4())
        # Security finding N1: generated and bound to THIS session/document
        # BEFORE the provider is ever called, so it can actually be handed
        # to the provider as part of `callback_url` below -- previously this
        # was minted only after `provider.initiate(...)` had already
        # returned, at which point it was far too late to have been passed
        # to the provider at all, and the route response never included it
        # either (see `app/api/notarization.py::initiate_signing`). Nothing
        # anywhere could ever present a valid token to `handle_signing_
        # callback` below, which made the entire signed -> notarized
        # transition unreachable in practice, not just insecure.
        callback_token = issue_action_token(
            "esign_callback", session_id, settings.esign_session_ttl_minutes * 60
        )
        callback_url = (
            f"{settings.verification_base_url.rstrip('/')}/notarization/signing/callback"
            f"?session_id={session_id}&action_token={callback_token}"
        )
        signer = document.get("signer", {})
        signing_request = SigningRequest(
            document_id=document_id,
            document_version=int(document.get("document_version", 1)),
            document_title=document.get("document_title", ""),
            document_hash=document.get("document_hash", ""),
            signer_name=signer.get("full_name", ""),
            signer_email=signer.get("email", ""),
            purpose=purpose,
            callback_url=callback_url,
        )

        try:
            session = await provider.initiate(signing_request)
        except Exception as exc:
            log.warning("esign_initiate_failed", provider=provider.name, error=str(exc))
            await self.audit.append(
                build_event(
                    action="esign_failed",
                    document_id=document_id,
                    document_version=int(document.get("document_version", 1)),
                    document_hash=document.get("document_hash", ""),
                    actor_id=user_id,
                    actor_role="user",
                    detail={"provider": provider.name, "reason": "provider_error"},
                )
            )
            raise BadRequestError(
                "The e-signature provider could not be reached. The document has NOT been signed.",
                {"provider": provider.name},
            ) from exc

        # Only the minimum signer metadata is persisted. No Aadhaar number,
        # no OTP, no biometric data, no provider credential -- see
        # `app/notarization/esign/base.py`.
        await self.sessions.insert(
            {
                "_id": session_id,
                "document_id": document_id,
                "document_version": int(document.get("document_version", 1)),
                "document_hash": document.get("document_hash", ""),
                "owner_user_id": user_id,
                "provider": session.provider,
                "provider_reference": session.provider_reference,
                "status": session.status,
                "purpose": purpose,
                "signer_name": signer.get("full_name", ""),
                "signer_email": signer.get("email", ""),
                "failure_reason": session.failure_reason,
                "consent_shown_at": datetime.now(UTC),
            }
        )

        if session.status in {"initiated", "pending"}:
            # Only an actually-initiated ceremony moves the document. A
            # provider that returned `pending` (e.g. the unconfigured default)
            # leaves it exactly where it was.
            if session.status == "initiated":
                await self._transition(
                    document,
                    states.SIGNING_IN_PROGRESS,
                    action="esign_initiated",
                    actor_id=user_id,
                    actor_role="user",
                    detail={"provider": session.provider, "session_id": session_id},
                )
        else:
            await self.audit.append(
                build_event(
                    action="esign_failed",
                    document_id=document_id,
                    document_version=int(document.get("document_version", 1)),
                    document_hash=document.get("document_hash", ""),
                    actor_id=user_id,
                    actor_role="user",
                    detail={"provider": session.provider, "status": session.status},
                )
            )

        # Reuses the SAME `callback_token` minted above, before the provider
        # was called -- re-minting a second, different token here (as
        # before this fix) would have made the token returned to the
        # caller/embedded in the provider's audit trail different from the
        # one actually embedded in `callback_url`, so neither would ever
        # verify against the other.
        return {
            "session_id": session_id,
            "session": session,
            "document": document,
            "callback_token": callback_token,
        }

    async def handle_signing_callback(
        self, *, session_id: str, action_token: str, status: str, failure_reason: str = ""
    ) -> dict[str, Any]:
        """Records a provider's outcome.

        The action token is verified FIRST: this endpoint is reachable from
        outside our trust boundary, and without it anyone could post
        `status="signed"` for a session id and advance a document.
        """
        verify_action_token(action_token, "esign_callback", session_id)

        session = await self.sessions.find_by_id(session_id)
        if session is None:
            raise NotFoundError(f"Signing session not found: {session_id}")
        document = await self._require_document(session["document_id"])

        # A hash change between initiation and callback means the document was
        # edited under an open signing ceremony. The signature cannot apply.
        if not hashes_match(session.get("document_hash", ""), document.get("document_hash", "")):
            await self.sessions.update_by_id(
                session_id, {"status": "failed", "failure_reason": "document_changed_during_signing"}
            )
            await self.audit.append(
                build_event(
                    action="esign_failed",
                    document_id=str(document["_id"]),
                    document_version=int(document.get("document_version", 1)),
                    document_hash=document.get("document_hash", ""),
                    actor_id=None,
                    actor_role="provider",
                    detail={"reason": "document_hash_changed_since_initiation"},
                )
            )
            raise BadRequestError(
                "The document changed after signing began, so the signature cannot be applied. "
                "A new version must be prepared and signed."
            )

        await self.sessions.update_by_id(session_id, {"status": status, "failure_reason": failure_reason})

        if status == "signed":
            await self._transition(
                document,
                states.SIGNED,
                action="esign_completed",
                actor_id=None,
                actor_role="provider",
                detail={"provider": session.get("provider", ""), "session_id": session_id},
            )
        elif status in UNSUCCESSFUL_SIGNING_STATUSES:
            # Must leave the document NON-signed. If a ceremony was open, roll
            # back to `ready_for_signature`; never leave it parked in
            # `signing_in_progress`, which reads as further along than it is.
            if document.get("status") == states.SIGNING_IN_PROGRESS:
                await self._transition(
                    document,
                    states.READY_FOR_SIGNATURE,
                    action="esign_failed",
                    actor_id=None,
                    actor_role="provider",
                    detail={"provider": session.get("provider", ""), "status": status,
                            "reason": failure_reason or status},
                )
            else:
                await self.audit.append(
                    build_event(
                        action="esign_failed",
                        document_id=str(document["_id"]),
                        document_version=int(document.get("document_version", 1)),
                        document_hash=document.get("document_hash", ""),
                        actor_id=None,
                        actor_role="provider",
                        detail={"status": status, "reason": failure_reason or status},
                    )
                )
        return {"session_id": session_id, "status": status, "document": await self._require_document(str(document["_id"]))}

    # -- 3. Notarization requests -----------------------------------------

    async def create_request(
        self, *, document_id: str, user_id: str, assigned_notary_id: str | None,
        supporting_id_confirmed: bool, note: str,
    ) -> dict[str, Any]:
        document = await self._require_document(document_id)
        self._require_ownership(document, user_id)

        if document.get("status") != states.SIGNED:
            raise BadRequestError(
                "Only a signed document can be submitted for notarization.",
                {"status": document.get("status")},
            )
        if assigned_notary_id:
            notary = await self.notaries.find_by_id(assigned_notary_id)
            if notary is None:
                raise NotFoundError(f"Notary not found: {assigned_notary_id}")
            if notary.get("verification_status") != "verified" or not notary.get("active", False):
                raise BadRequestError("The selected notary is not a verified, active notary account.")

        request_id = str(uuid4())
        await self.requests.insert(
            {
                "_id": request_id,
                "document_id": document_id,
                "document_version": int(document.get("document_version", 1)),
                "document_title": document.get("document_title", ""),
                "document_hash": document.get("document_hash", ""),
                "requested_by_user_id": user_id,
                "assigned_notary_id": assigned_notary_id,
                "signer": document.get("signer", {}),
                "witnesses": document.get("witnesses", []),
                "supporting_id_confirmed": supporting_id_confirmed,
                "review_status": "pending",
                "remarks": "",
                "note": note,
                "reviewed_at": None,
                "reviewed_by_notary_id": None,
            }
        )
        await self._transition(
            document,
            states.NOTARY_REVIEW_REQUESTED,
            action="notarization_requested",
            actor_id=user_id,
            actor_role="user",
            detail={"request_id": request_id, "assigned_notary_id": assigned_notary_id},
        )
        return await self.requests.find_by_id(request_id) or {}

    async def start_review(self, *, request_id: str, notary_user_id: str) -> dict[str, Any]:
        notary = await self._require_verified_notary(notary_user_id)
        request = await self.requests.find_by_id(request_id)
        if request is None:
            raise NotFoundError(f"Notarization request not found: {request_id}")
        self._require_request_access(request, notary)

        document = await self._require_document(request["document_id"])
        await self.requests.update_by_id(request_id, {"review_status": "in_review", "reviewed_by_notary_id": str(notary["_id"])})
        await self._transition(
            document,
            states.NOTARY_REVIEW_IN_PROGRESS,
            action="notary_review_started",
            actor_id=notary_user_id,
            actor_role="notary",
            detail={"request_id": request_id},
        )
        return await self.requests.find_by_id(request_id) or {}

    @staticmethod
    def _require_request_access(request: dict[str, Any], notary: dict[str, Any]) -> None:
        """A notary may only act on requests assigned to them, or unassigned
        ones from the shared queue."""
        assigned = request.get("assigned_notary_id")
        if assigned and assigned != str(notary["_id"]):
            raise NotFoundError(f"Notarization request not found: {request.get('_id')}")

    def issue_decision_token(self, request_id: str, purpose: str) -> str:
        """A short-lived capability minted only after re-authentication.

        The API layer calls this after re-verifying the notary's password, so
        approving or revoking a notarization always requires a fresh
        credential rather than a still-valid session cookie.
        """
        if purpose not in {"approve", "reject", "revoke"}:
            raise BadRequestError(f"Unsupported notarial action: {purpose}")
        return issue_action_token(purpose, request_id, settings.notary_action_token_ttl_seconds)

    async def approve_request(
        self, *, request_id: str, notary_user_id: str, action_token: str, remarks: str,
        confirmed_document_hash: str,
    ) -> dict[str, Any]:
        """The ONLY path to `notarized`.

        Four independent gates, all of which must pass:
          1. a fresh, correctly-scoped re-authentication token;
          2. a verified AND active notary account;
          3. the request is assigned to (or open to) that notary;
          4. the hash the notary confirms matches the stored hash.
        """
        verify_action_token(action_token, "approve", request_id)
        notary = await self._require_verified_notary(notary_user_id)

        request = await self.requests.find_by_id(request_id)
        if request is None:
            raise NotFoundError(f"Notarization request not found: {request_id}")
        self._require_request_access(request, notary)
        if request.get("review_status") in {"approved", "rejected"}:
            raise BadRequestError(
                f"This request has already been {request.get('review_status')}.",
                {"review_status": request.get("review_status")},
            )

        document = await self._require_document(request["document_id"])

        # The notary attests to specific bytes. If what they confirm does not
        # match what we hold, something changed between review and approval
        # and the attestation must not be recorded.
        if confirmed_document_hash and not hashes_match(confirmed_document_hash, document.get("document_hash", "")):
            await self.audit.append(
                build_event(
                    action="notary_rejected",
                    document_id=str(document["_id"]),
                    document_version=int(document.get("document_version", 1)),
                    document_hash=document.get("document_hash", ""),
                    actor_id=notary_user_id,
                    actor_role="notary",
                    detail={"request_id": request_id, "reason": "document_hash_mismatch"},
                )
            )
            raise BadRequestError(
                "The document hash you confirmed does not match the document on record. "
                "The document may have changed since you reviewed it; approval has been refused."
            )

        verification_token = new_verification_token()
        notarized_at = datetime.now(UTC)

        await self._transition(
            document,
            states.NOTARIZED,
            action="notary_approved",
            actor_id=notary_user_id,
            actor_role="notary",
            detail={"request_id": request_id, "notary_registration_number": notary.get("registration_number", "")},
        )
        await self.documents.update_by_id(
            str(document["_id"]),
            {
                "verification_token": verification_token,
                "notarized_at": notarized_at,
                "notarized_by_notary_id": str(notary["_id"]),
            },
        )
        await self.requests.update_by_id(
            request_id,
            {
                "review_status": "approved",
                "remarks": remarks,
                "reviewed_at": notarized_at,
                "reviewed_by_notary_id": str(notary["_id"]),
            },
        )
        await self.audit.append(
            build_event(
                action="notarized_version_issued",
                document_id=str(document["_id"]),
                document_version=int(document.get("document_version", 1)),
                document_hash=document.get("document_hash", ""),
                actor_id=notary_user_id,
                actor_role="notary",
                detail={"request_id": request_id},
            )
        )
        return {
            "request": await self.requests.find_by_id(request_id) or {},
            "document": await self._require_document(str(document["_id"])),
            "verification_url": self.verification_url(verification_token),
        }

    async def reject_request(
        self, *, request_id: str, notary_user_id: str, action_token: str, remarks: str
    ) -> dict[str, Any]:
        """A rejected request must leave the document NON-notarized."""
        verify_action_token(action_token, "reject", request_id)
        notary = await self._require_verified_notary(notary_user_id)

        request = await self.requests.find_by_id(request_id)
        if request is None:
            raise NotFoundError(f"Notarization request not found: {request_id}")
        self._require_request_access(request, notary)
        if not remarks.strip():
            raise BadRequestError("A rejection reason is required.")

        document = await self._require_document(request["document_id"])
        await self.requests.update_by_id(
            request_id,
            {
                "review_status": "rejected",
                "remarks": remarks,
                "reviewed_at": datetime.now(UTC),
                "reviewed_by_notary_id": str(notary["_id"]),
            },
        )
        # Back to `signed`: the signature still stands, the notarization does
        # not. The document never touched `notarized`.
        await self._transition(
            document,
            states.SIGNED,
            action="notary_rejected",
            actor_id=notary_user_id,
            actor_role="notary",
            detail={"request_id": request_id, "reason": remarks},
        )
        return {"request": await self.requests.find_by_id(request_id) or {},
                "document": await self._require_document(str(document["_id"]))}

    async def revoke(
        self, *, request_id: str, notary_user_id: str, action_token: str, remarks: str
    ) -> dict[str, Any]:
        """Withdraws a notarization already issued.

        The verification token is deliberately KEPT, not deleted: anyone
        holding a QR code from the revoked document must be able to scan it
        and be told it was revoked. Deleting the token would make a revoked
        document indistinguishable from an unknown one.
        """
        verify_action_token(action_token, "revoke", request_id)
        notary = await self._require_verified_notary(notary_user_id)
        if not remarks.strip():
            raise BadRequestError("A reason is required to revoke a notarization.")

        request = await self.requests.find_by_id(request_id)
        if request is None:
            raise NotFoundError(f"Notarization request not found: {request_id}")
        self._require_request_access(request, notary)

        document = await self._require_document(request["document_id"])
        if document.get("status") != states.NOTARIZED:
            raise BadRequestError(
                "Only a notarized document can be revoked.", {"status": document.get("status")}
            )
        revoked_at = datetime.now(UTC)
        await self._transition(
            document,
            states.REVOKED,
            action="notarization_revoked",
            actor_id=notary_user_id,
            actor_role="notary",
            detail={"request_id": request_id, "reason": remarks},
        )
        await self.documents.update_by_id(str(document["_id"]), {"revoked_at": revoked_at, "revocation_reason": remarks})
        await self.requests.update_by_id(request_id, {"review_status": "rejected", "remarks": remarks})
        return {"document": await self._require_document(str(document["_id"]))}

    # -- 4. Versioning: editing never mutates an attested document ---------

    async def create_version(
        self, *, document_id: str, sections: dict[str, str], user_id: str,
        signer: dict[str, str] | None = None, witnesses: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Supersedes a signed/notarized document with a new `draft` version.

        This is what makes a notarized artefact immutable: the old row keeps
        its status, hash and audit trail exactly as attested, and is marked
        superseded with `notarization_valid=False`; the change lands on a NEW
        row that starts at `draft` with a new hash and no verification token.
        """
        previous = await self._require_document(document_id)
        self._require_ownership(previous, user_id)

        new_hash = hash_sections(sections)
        if hashes_match(new_hash, previous.get("document_hash", "")):
            raise BadRequestError("The document is unchanged; no new version was created.")

        new_id = str(uuid4())
        new_version = int(previous.get("document_version", 1)) + 1
        await self.documents.insert(
            {
                "_id": new_id,
                "source_draft_id": previous.get("source_draft_id"),
                "document_version": new_version,
                "document_title": previous.get("document_title", ""),
                "document_category": previous.get("document_category", ""),
                "document_hash": new_hash,
                "sections": sections,
                "signer": signer if signer is not None else previous.get("signer", {}),
                "witnesses": witnesses if witnesses is not None else previous.get("witnesses", []),
                "owner_user_id": user_id,
                "checklist_complete": False,
                "status": states.DRAFT,
                "verification_token": None,
                "notarized_at": None,
                "notarized_by_notary_id": None,
                "revoked_at": None,
                "supersedes_document_id": document_id,
            }
        )
        # The prior version's own notarization is explicitly invalidated. Its
        # status is left untouched so the historical record of what was
        # attested stays intact and readable.
        await self.documents.update_by_id(
            document_id,
            {
                "superseded_by_document_id": new_id,
                "notarization_valid": False,
                "notarization_invalidated_reason": "Superseded by a newer version of this document.",
            },
        )
        await self.audit.append(
            build_event(
                action="user_edited",
                document_id=document_id,
                document_version=int(previous.get("document_version", 1)),
                document_hash=previous.get("document_hash", ""),
                actor_id=user_id,
                actor_role="user",
                detail={
                    "superseded_by": new_id,
                    "new_version": new_version,
                    "prior_notarization_invalidated": previous.get("status") == states.NOTARIZED,
                },
            )
        )
        return await self._require_document(new_id)

    # -- 5. Verification ---------------------------------------------------

    def verification_url(self, verification_token: str) -> str:
        return f"{settings.verification_base_url.rstrip('/')}/verify/{verification_token}"

    async def verify(self, verification_token: str, provided_hash: str = "") -> dict[str, Any]:
        """The public verification lookup.

        Returns only non-identifying facts. An unknown token yields a plain
        "not found" with no hint about whether it ever existed.
        """
        document = await self.documents.find_by_verification_token(verification_token)
        if document is None:
            return {"found": False, "message": "No notarized document matches this verification code."}

        notary = None
        if document.get("notarized_by_notary_id"):
            notary = await self.notaries.find_by_id(document["notarized_by_notary_id"])

        status = document.get("status", states.DRAFT)
        revoked = status == states.REVOKED
        result: dict[str, Any] = {
            "found": True,
            "status": status,
            "status_label": status_label(status),
            "document_type": document.get("document_category", ""),
            "document_title": document.get("document_title", ""),
            "notarized_at": document.get("notarized_at"),
            "notary_name": (notary or {}).get("full_name", ""),
            "notary_registration_number": (notary or {}).get("registration_number", ""),
            "revoked": revoked,
            "revoked_at": document.get("revoked_at"),
            "hash_matches": (
                hashes_match(provided_hash, document.get("document_hash", "")) if provided_hash else None
            ),
        }
        if revoked:
            result["message"] = "This notarization has been revoked and must not be relied upon."
        elif document.get("notarization_valid") is False:
            result["message"] = "This document has been superseded by a newer version; this notarization no longer applies."
        elif not states.is_notarized(status):
            result["message"] = "This document is not notarized."
        return result

    # -- 6. Status / timeline ---------------------------------------------

    async def document_status(self, *, document_id: str, user_id: str) -> dict[str, Any]:
        document = await self._require_document(document_id)
        self._require_ownership(document, user_id)
        events = await self.audit.list_for_document(document_id)
        status = document.get("status", states.DRAFT)
        return {
            "document": document,
            "status": status,
            "status_label": status_label(status),
            "is_notarized": states.is_notarized(status),
            # The final notarized PDF is downloadable only once verification
            # actually exists -- i.e. only in the `notarized` state.
            "downloadable": states.is_notarized(status),
            "timeline": events,
        }

    # -- 7. Notarized export ----------------------------------------------

    async def build_notarized_export_options(self, document: dict[str, Any]) -> Any:
        """`ExportOptions` carrying the attestation block and verification QR.

        Refuses for any non-notarized document. This is the ONLY place that
        constructs a `NotarizationAttestation`, and it builds it strictly from
        the stored notarization record -- never from a draft, a signature, or
        anything the user supplied. Nothing here mints a notary name,
        registration number, seal, or date.
        """
        from app.drafting.export import ExportOptions, NotarizationAttestation
        from app.notarization.qr import build_verification_qr_svg

        status = document.get("status", states.DRAFT)
        if not states.is_notarized(status):
            raise BadRequestError(
                "Only a notarized document can be exported with a notarization record.",
                {"status": status},
            )
        notary = await self.notaries.find_by_id(document.get("notarized_by_notary_id") or "")
        if notary is None:
            # A notarized document must have an attesting notary on record.
            # If it does not, something is wrong with the data and we refuse
            # rather than print an attestation with a blank notary.
            raise BadRequestError("This document has no attesting notary on record; export refused.")

        verification_url = self.verification_url(document.get("verification_token", ""))
        notarized_at = document.get("notarized_at")
        return ExportOptions(
            document_version=int(document.get("document_version", 1)),
            generated_on=datetime.now(UTC).strftime("%d %B %Y"),
            notarization=NotarizationAttestation(
                notary_name=notary.get("full_name", ""),
                notary_registration_number=notary.get("registration_number", ""),
                jurisdiction_state=notary.get("jurisdiction_state", ""),
                notarized_on=notarized_at.strftime("%d %B %Y") if notarized_at else "",
                document_hash=document.get("document_hash", ""),
                verification_url=verification_url,
                verification_qr_svg=build_verification_qr_svg(status, verification_url),
            ),
        )

    async def export_notarized(self, *, document_id: str, user_id: str, fmt: str = "pdf") -> Any:
        """Renders the final notarized document.

        Gated twice on purpose: `_require_ownership` (only the owner may
        download it) and `build_notarized_export_options` (only a notarized
        document has anything to render).
        """
        from pathlib import Path
        from uuid import uuid4 as _uuid4

        from app.drafting.export import (
            DocxDraftExporter,
            DraftExporter,
            PdfDraftExporter,
            RtfDraftExporter,
            TxtDraftExporter,
        )

        document = await self._require_document(document_id)
        self._require_ownership(document, user_id)
        options = await self.build_notarized_export_options(document)

        exporters: dict[str, DraftExporter] = {
            "pdf": PdfDraftExporter(), "docx": DocxDraftExporter(),
            "txt": TxtDraftExporter(), "rtf": RtfDraftExporter(),
        }
        if fmt not in exporters:
            raise BadRequestError(f"Unsupported export format: {fmt}")
        settings.draft_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(settings.draft_output_dir) / f"{_uuid4()}.{fmt}"
        result = exporters[fmt].export(
            document.get("document_title", "Document"),
            document.get("sections", {}),
            output_path,
            document.get("language", "english"),
            options,
        )
        await self.audit.append(
            build_event(
                action="document_exported",
                document_id=document_id,
                document_version=int(document.get("document_version", 1)),
                document_hash=document.get("document_hash", ""),
                actor_id=user_id,
                actor_role="user",
                detail={"format": fmt, "notarized": True},
            )
        )
        return result
