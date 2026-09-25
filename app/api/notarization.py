"""E-Notarization API.

Route grouping mirrors the trust boundaries:

* `/notarization/*`            -- authenticated document owner
* `/notarization/requests/*`   -- verified notary (approve/reject/revoke)
* `/notarization/admin/*`      -- admin (notary account verification, audit)
* `/verify/{token}`            -- PUBLIC, unauthenticated, rate-limited

Every document-scoped handler goes through `NotarizationService`, which owns
the ownership checks and the state machine. Nothing here decides on its own
whether a document is notarized.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from app.api.deps import get_current_user_claims, get_current_user_id, require_admin
from app.cache.redis_client import redis_client
from app.core.config import settings
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError, RateLimitError
from app.core.security import verify_password
from app.drafting.engine import LegalDraftEngine
from app.drafting.templates import get_template
from app.notarization import states
from app.notarization.audit import build_event
from app.notarization.checklist import evaluate as evaluate_checklist
from app.notarization.esign import available_providers
from app.notarization.service import NotarizationService, status_label
from app.repositories.notarization import (
    ESignSessionRepository,
    NotarizationAuditRepository,
    NotaryAccountRepository,
)
from app.repositories.users import UserRepository
from app.schemas.notarization import (
    STANDARD_WARNINGS,
    ChecklistItemModel,
    CreateNotarizationRequestPayload,
    DocumentStatusResponse,
    DocumentStatusTimelineEntry,
    InitiateSigningRequest,
    InitiateSigningResponse,
    NotarizationRequestSummary,
    NotaryAccountPayload,
    NotaryAccountSummary,
    NotaryDecisionRequest,
    NotaryDecisionResponse,
    PrepareNotarizationRequest,
    PrepareNotarizationResponse,
    PublicVerificationResponse,
    SigningCallbackRequest,
    SigningConsent,
)
from app.utils.request_ip import client_ip

router = APIRouter(tags=["notarization"])

_service = NotarizationService()


# ---------------------------------------------------------------------------
# Flow-specific rate limits.
#
# The global `RateLimitMiddleware` protects the API as a whole. These two
# flows need their own, tighter ceilings for different reasons: `/verify` is
# unauthenticated and its tokens are in principle enumerable, and the signing
# endpoints trigger outbound vendor calls that cost money and can be used to
# spam a signer's inbox. Fails OPEN on a Redis outage, matching the existing
# middleware -- protective infrastructure must not take the API down with it.
# ---------------------------------------------------------------------------
async def _enforce_rate_limit(request: Request, bucket: str, limit_per_minute: int) -> None:
    key = f"notarization-rate-limit:{bucket}:{client_ip(request)}:{int(time.time() // 60)}"
    try:
        count = await redis_client.client.incr(key)
        if count == 1:
            await redis_client.client.expire(key, 60)
        if count > limit_per_minute:
            raise RateLimitError("Too many requests. Please try again shortly.")
    except RateLimitError:
        raise
    except Exception:  # noqa: BLE001 - a cache outage must not break the flow
        return


async def _require_reauthentication(user_id: str, password: str) -> None:
    """Fresh credential check before a notarial act.

    Approving, rejecting or revoking a notarization is not something a
    still-valid session should be able to do on its own -- a borrowed laptop
    would be enough. The password is verified here and never stored, logged,
    or echoed.
    """
    if not password:
        raise ForbiddenError("Re-authentication is required for this action.")
    user = await UserRepository().find_by_id(user_id)
    if user is None or not verify_password(password, user.get("password_hash", "")):
        raise ForbiddenError("Re-authentication failed.")


# ---------------------------------------------------------------------------
# 1. Prepare for notarization
# ---------------------------------------------------------------------------


@router.post("/notarization/prepare", response_model=PrepareNotarizationResponse)
async def prepare_for_notarization(
    payload: PrepareNotarizationRequest,
    user_id: str | None = Depends(get_current_user_id),
) -> PrepareNotarizationResponse:
    if not user_id:
        raise ForbiddenError("Authentication is required to prepare a document for notarization.")

    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(payload.draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {payload.draft_id}")
    if draft.get("user_id") and draft.get("user_id") != user_id:
        raise NotFoundError(f"Draft not found: {payload.draft_id}")

    template = get_template(draft["draft_type"])
    if template is None:
        raise NotFoundError(f"Unknown draft template: {draft['draft_type']}")

    result = await _service.prepare(
        draft_id=payload.draft_id,
        draft_title=draft.get("template_name", template.name),
        draft_category=template.category,
        sections=draft.get("sections", {}),
        signer=payload.signer.model_dump(),
        witnesses=[witness.model_dump() for witness in payload.witnesses],
        user_id=user_id,
    )
    readiness = result["readiness"]
    document = result["document"]
    return PrepareNotarizationResponse(
        document_id=str(document["_id"]),
        document_version=result["document_version"],
        document_hash=result["document_hash"],
        status=document["status"],
        eligible=readiness.eligible,
        checklist_complete=readiness.complete,
        checklist=[ChecklistItemModel(**item) for item in readiness.as_dicts()],
        label=readiness.label,
        ineligible_reason=readiness.ineligible_reason,
    )


@router.get("/notarization/eligibility")
async def notarization_eligibility(
    draft_id: str = Query(min_length=1),
    user_id: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """Whether a draft type can be prepared for notarization at all, so the UI
    can decide whether to offer the button.

    Security finding N3: this previously took no authentication and no
    ownership check at all, letting any caller -- even an unauthenticated
    one -- probe an arbitrary `draft_id` for its category/type. Same gate as
    `POST /notarization/prepare` immediately above: authentication is
    required, and a draft owned by a DIFFERENT account reports 404 rather
    than 403, so this can't be used to distinguish "not yours" from
    "doesn't exist" either.
    """
    if not user_id:
        raise ForbiddenError("Authentication is required to check notarization eligibility.")
    engine = LegalDraftEngine()
    draft = await engine.drafts.find_by_id(draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {draft_id}")
    if draft.get("user_id") and draft.get("user_id") != user_id:
        raise NotFoundError(f"Draft not found: {draft_id}")
    template = get_template(draft["draft_type"])
    if template is None:
        raise NotFoundError(f"Unknown draft template: {draft['draft_type']}")
    readiness = evaluate_checklist(template.category, {}, [])
    return {
        "draft_id": draft_id,
        "category": template.category,
        "eligible": readiness.eligible,
        "reason": readiness.ineligible_reason,
        "warnings": list(STANDARD_WARNINGS),
    }


# ---------------------------------------------------------------------------
# 2. E-signature
# ---------------------------------------------------------------------------


@router.get("/notarization/signing/providers")
async def list_signing_providers(_claims: dict[str, Any] = Depends(get_current_user_claims)) -> dict[str, Any]:
    return {"providers": available_providers(), "configured": settings.esign_provider}


@router.post("/notarization/signing/initiate", response_model=InitiateSigningResponse)
async def initiate_signing(
    request: Request,
    payload: InitiateSigningRequest,
    user_id: str | None = Depends(get_current_user_id),
) -> InitiateSigningResponse:
    if not user_id:
        raise ForbiddenError("Authentication is required.")
    await _enforce_rate_limit(request, "signing", settings.signing_rate_limit_per_minute)

    result = await _service.initiate_signing(
        document_id=payload.document_id, purpose=payload.purpose, user_id=user_id,
        provider_name=payload.provider,
    )
    session = result["session"]
    document = result["document"]
    from datetime import UTC, datetime

    return InitiateSigningResponse(
        session_id=result["session_id"],
        document_id=payload.document_id,
        provider=session.provider,
        status=session.status,
        signing_url=session.signing_url,
        # The consent panel the user must see BEFORE signing.
        consent=SigningConsent(
            document_title=document.get("document_title", ""),
            document_hash=document.get("document_hash", ""),
            signer_name=document.get("signer", {}).get("full_name", ""),
            purpose=payload.purpose,
            signing_timestamp=datetime.now(UTC),
        ),
        message=session.failure_reason,
        # Security finding N1: this was being computed and returned by the
        # service (`result["callback_token"]`) and silently dropped here --
        # see `InitiateSigningResponse.callback_token`'s own docstring.
        callback_token=result["callback_token"],
    )


@router.post("/notarization/signing/callback")
async def signing_callback(request: Request, payload: SigningCallbackRequest) -> dict[str, Any]:
    """Provider callback. Unauthenticated by necessity, but every call must
    carry the signed action token issued when the session was created."""
    await _enforce_rate_limit(request, "signing-callback", settings.signing_rate_limit_per_minute)
    result = await _service.handle_signing_callback(
        session_id=payload.session_id,
        action_token=payload.action_token,
        status=payload.status,
        failure_reason=payload.failure_reason,
    )
    document = result["document"]
    return {
        "session_id": result["session_id"],
        "status": result["status"],
        "document_status": document.get("status"),
        "status_label": status_label(document.get("status", states.DRAFT)),
        # Said explicitly on every callback: a signature is not a notarization.
        "notice": "Digital signing alone may not constitute notarization.",
    }


# ---------------------------------------------------------------------------
# 3. Notarization requests
# ---------------------------------------------------------------------------


@router.post("/notarization/requests", response_model=NotarizationRequestSummary)
async def create_notarization_request(
    payload: CreateNotarizationRequestPayload,
    user_id: str | None = Depends(get_current_user_id),
) -> NotarizationRequestSummary:
    if not user_id:
        raise ForbiddenError("Authentication is required.")
    request_doc = await _service.create_request(
        document_id=payload.document_id,
        user_id=user_id,
        assigned_notary_id=payload.assigned_notary_id,
        supporting_id_confirmed=payload.supporting_id_confirmed,
        note=payload.note,
    )
    return _request_summary(request_doc)


@router.get("/notarization/requests", response_model=list[NotarizationRequestSummary])
async def list_notarization_requests(
    claims: dict[str, Any] = Depends(get_current_user_claims),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[NotarizationRequestSummary]:
    """The caller's own requests, or -- for a verified notary -- their queue.

    A user never sees another user's requests, and a notary never sees a
    request assigned to a different notary.
    """
    user_id = claims.get("sub")
    notary = await NotaryAccountRepository().find_by_user_id(user_id) if user_id else None
    if notary and notary.get("verification_status") == "verified" and notary.get("active"):
        assigned = await _service.requests.list_for_notary(str(notary["_id"]), status, limit)
        unassigned = [
            request
            for request in await _service.requests.list_pending(limit)
            if not request.get("assigned_notary_id")
        ]
        seen: set[str] = set()
        merged = []
        for request in [*assigned, *unassigned]:
            if str(request["_id"]) not in seen:
                seen.add(str(request["_id"]))
                merged.append(request)
        return [_request_summary(request) for request in merged[:limit]]
    return [_request_summary(request) for request in await _service.requests.list_for_requester(user_id or "", limit)]


@router.post("/notarization/requests/{request_id}/start-review", response_model=NotarizationRequestSummary)
async def start_review(request_id: str, claims: dict[str, Any] = Depends(get_current_user_claims)) -> NotarizationRequestSummary:
    return _request_summary(await _service.start_review(request_id=request_id, notary_user_id=claims.get("sub", "")))


@router.post("/notarization/requests/{request_id}/decision-token")
async def mint_decision_token(
    request_id: str,
    action: str = Query(pattern="^(approve|reject|revoke)$"),
    password: str = Query(min_length=1),
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> dict[str, Any]:
    """Re-authenticate, then mint a short-lived token for one notarial act.

    Separate from the act itself so the password is submitted exactly once,
    to one endpoint, and the resulting capability is narrowly scoped to this
    request id and this action.
    """
    user_id = claims.get("sub", "")
    await _require_reauthentication(user_id, password)
    await _service._require_verified_notary(user_id)
    return {
        "action_token": _service.issue_decision_token(request_id, action),
        "expires_in_seconds": settings.notary_action_token_ttl_seconds,
    }


@router.post("/notarization/requests/{request_id}/approve", response_model=NotaryDecisionResponse)
async def approve_request(
    request_id: str, payload: NotaryDecisionRequest, claims: dict[str, Any] = Depends(get_current_user_claims)
) -> NotaryDecisionResponse:
    result = await _service.approve_request(
        request_id=request_id,
        notary_user_id=claims.get("sub", ""),
        action_token=payload.action_token,
        remarks=payload.remarks,
        confirmed_document_hash=payload.confirmed_document_hash,
    )
    document = result["document"]
    return NotaryDecisionResponse(
        request_id=request_id,
        document_id=str(document["_id"]),
        review_status="approved",
        document_status=document["status"],
        verification_url=result["verification_url"],
        message="The document has been notarized and a verification code issued.",
    )


@router.post("/notarization/requests/{request_id}/reject", response_model=NotaryDecisionResponse)
async def reject_request(
    request_id: str, payload: NotaryDecisionRequest, claims: dict[str, Any] = Depends(get_current_user_claims)
) -> NotaryDecisionResponse:
    result = await _service.reject_request(
        request_id=request_id,
        notary_user_id=claims.get("sub", ""),
        action_token=payload.action_token,
        remarks=payload.remarks,
    )
    document = result["document"]
    return NotaryDecisionResponse(
        request_id=request_id,
        document_id=str(document["_id"]),
        review_status="rejected",
        document_status=document["status"],
        message="The request was rejected. The document remains NOT notarized.",
    )


@router.post("/notarization/requests/{request_id}/revoke", response_model=NotaryDecisionResponse)
async def revoke_notarization(
    request_id: str, payload: NotaryDecisionRequest, claims: dict[str, Any] = Depends(get_current_user_claims)
) -> NotaryDecisionResponse:
    result = await _service.revoke(
        request_id=request_id,
        notary_user_id=claims.get("sub", ""),
        action_token=payload.action_token,
        remarks=payload.remarks,
    )
    document = result["document"]
    return NotaryDecisionResponse(
        request_id=request_id,
        document_id=str(document["_id"]),
        review_status="revoked",
        document_status=document["status"],
        message="The notarization has been revoked and will show as revoked on verification.",
    )


# ---------------------------------------------------------------------------
# 4. Document status / timeline
# ---------------------------------------------------------------------------


@router.get("/notarization/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def document_status(document_id: str, user_id: str | None = Depends(get_current_user_id)) -> DocumentStatusResponse:
    if not user_id:
        raise ForbiddenError("Authentication is required.")
    result = await _service.document_status(document_id=document_id, user_id=user_id)
    document = result["document"]
    return DocumentStatusResponse(
        document_id=document_id,
        document_version=int(document.get("document_version", 1)),
        document_title=document.get("document_title", ""),
        status=result["status"],
        status_label=result["status_label"],
        document_hash=document.get("document_hash", ""),
        is_notarized=result["is_notarized"],
        downloadable=result["downloadable"],
        timeline=[
            DocumentStatusTimelineEntry(
                action=event["action"],
                actor_role=event.get("actor_role", "system"),
                document_version=int(event.get("document_version", 1)),
                occurred_at=event["occurred_at"],
                detail=event.get("detail", {}),
            )
            for event in result["timeline"]
        ],
    )


@router.get("/notarization/documents/{document_id}/download")
async def download_notarized_document(
    document_id: str,
    fmt: str = Query(default="pdf", pattern="^(pdf|docx|txt|rtf)$"),
    user_id: str | None = Depends(get_current_user_id),
) -> FileResponse:
    """The final notarized document.

    Available ONLY once the document is actually notarized -- the service
    refuses any other status. Before that point the user downloads the
    ordinary draft through `/draft/export`, which carries the AI-generated
    draft disclaimer and no notarization record.
    """
    if not user_id:
        raise ForbiddenError("Authentication is required.")
    path = await _service.export_notarized(document_id=document_id, user_id=user_id, fmt=fmt)
    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "rtf": "application/rtf",
    }
    return FileResponse(path, media_type=media_types[fmt], filename=f"notarized-document.{fmt}")


# ---------------------------------------------------------------------------
# 5. Public verification
# ---------------------------------------------------------------------------


@router.get("/verify/{verification_token}", response_model=PublicVerificationResponse)
async def verify_document(
    request: Request,
    verification_token: str,
    document_hash: str = Query(default="", max_length=64),
) -> PublicVerificationResponse:
    """PUBLIC. Returns only the fields on `PublicVerificationResponse`.

    Never exposes document content, signer contact/address details, identity
    document information, or the owning user's identity.
    """
    await _enforce_rate_limit(request, "verify", settings.verification_rate_limit_per_minute)
    result = await _service.verify(verification_token, document_hash)
    return PublicVerificationResponse(**result)


# ---------------------------------------------------------------------------
# 6. Admin
# ---------------------------------------------------------------------------

admin_router = APIRouter(tags=["notarization-admin"], dependencies=[Depends(require_admin)])


@admin_router.post("/notarization/admin/notaries", response_model=NotaryAccountSummary)
async def register_notary(payload: NotaryAccountPayload) -> NotaryAccountSummary:
    """Creates a notary account in the UNVERIFIED state.

    Registration and verification are deliberately two steps: creating an
    account must never, by itself, confer the power to notarize.
    """
    repository = NotaryAccountRepository()
    if await repository.find_by_registration_number(payload.registration_number):
        raise BadRequestError("A notary with this registration number already exists.")
    notary_id = await repository.insert(
        {
            "user_id": payload.user_id,
            "full_name": payload.full_name,
            "registration_number": payload.registration_number,
            "jurisdiction_state": payload.jurisdiction_state,
            "certificate_reference": payload.certificate_reference,
            "verification_status": "unverified",
            "active": False,
            "verified_by_admin_id": None,
            "verified_at": None,
        }
    )
    return _notary_summary(await repository.find_by_id(notary_id) or {})


@admin_router.post("/notarization/admin/notaries/{notary_id}/verify", response_model=NotaryAccountSummary)
async def verify_notary(notary_id: str, claims: dict[str, Any] = Depends(get_current_user_claims)) -> NotaryAccountSummary:
    from datetime import UTC, datetime

    repository = NotaryAccountRepository()
    notary = await repository.find_by_id(notary_id)
    if notary is None:
        raise NotFoundError(f"Notary not found: {notary_id}")
    await repository.update_by_id(
        notary_id,
        {
            "verification_status": "verified",
            "active": True,
            "verified_by_admin_id": claims.get("sub"),
            "verified_at": datetime.now(UTC),
        },
    )
    await NotarizationAuditRepository().append(
        build_event(
            action="notary_account_verified",
            document_id="",
            document_version=0,
            actor_id=claims.get("sub"),
            actor_role="admin",
            detail={"notary_id": notary_id, "registration_number": notary.get("registration_number", "")},
        )
    )
    return _notary_summary(await repository.find_by_id(notary_id) or {})


@admin_router.post("/notarization/admin/notaries/{notary_id}/revoke", response_model=NotaryAccountSummary)
async def revoke_notary(
    notary_id: str, reason: str = Query(min_length=1), claims: dict[str, Any] = Depends(get_current_user_claims)
) -> NotaryAccountSummary:
    repository = NotaryAccountRepository()
    if await repository.find_by_id(notary_id) is None:
        raise NotFoundError(f"Notary not found: {notary_id}")
    await repository.update_by_id(notary_id, {"verification_status": "revoked", "active": False})
    await NotarizationAuditRepository().append(
        build_event(
            action="notary_account_revoked",
            document_id="",
            document_version=0,
            actor_id=claims.get("sub"),
            actor_role="admin",
            detail={"notary_id": notary_id, "reason": reason},
        )
    )
    return _notary_summary(await repository.find_by_id(notary_id) or {})


@admin_router.get("/notarization/admin/notaries", response_model=list[NotaryAccountSummary])
async def list_notaries(limit: int = Query(default=100, ge=1, le=500)) -> list[NotaryAccountSummary]:
    return [_notary_summary(notary) for notary in await NotaryAccountRepository().list_all(limit)]


@admin_router.get("/notarization/admin/audit")
async def notarization_audit(
    document_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    repository = NotarizationAuditRepository()
    events = (
        await repository.list_for_document(document_id, limit)
        if document_id
        else await repository.list_recent(limit, action)
    )
    return {"events": events, "count": len(events)}


@admin_router.get("/notarization/admin/failed-attempts")
async def failed_attempts(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    """Failed/expired/cancelled signature sessions plus rejected notarization
    requests -- the review surface for suspicious activity."""
    sessions = await ESignSessionRepository().list_failed(limit)
    rejected = await NotarizationAuditRepository().list_recent(limit, "notary_rejected")
    return {
        "failed_signing_sessions": [
            {
                "session_id": str(session["_id"]),
                "document_id": session.get("document_id"),
                "provider": session.get("provider"),
                "status": session.get("status"),
                "failure_reason": session.get("failure_reason", ""),
                "created_at": session.get("created_at"),
            }
            for session in sessions
        ],
        "rejected_notarizations": rejected,
    }


@admin_router.get("/notarization/admin/esign-config")
async def esign_config() -> dict[str, Any]:
    """Provider configuration. Deliberately reports only NAMES and whether a
    credential is present -- never the credential itself."""
    return {
        "configured_provider": settings.esign_provider,
        "registered_providers": available_providers(),
        "session_ttl_minutes": settings.esign_session_ttl_minutes,
        "signing_rate_limit_per_minute": settings.signing_rate_limit_per_minute,
        "verification_rate_limit_per_minute": settings.verification_rate_limit_per_minute,
    }


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------


def _request_summary(request: dict[str, Any]) -> NotarizationRequestSummary:
    return NotarizationRequestSummary(
        request_id=str(request.get("_id", "")),
        document_id=request.get("document_id", ""),
        document_version=int(request.get("document_version", 1)),
        document_title=request.get("document_title", ""),
        document_hash=request.get("document_hash", ""),
        review_status=request.get("review_status", "pending"),
        signer_name=request.get("signer", {}).get("full_name", ""),
        supporting_id_confirmed=bool(request.get("supporting_id_confirmed", False)),
        witness_count=len(request.get("witnesses", [])),
        assigned_notary_id=request.get("assigned_notary_id"),
        remarks=request.get("remarks", ""),
        created_at=request.get("created_at"),
        reviewed_at=request.get("reviewed_at"),
    )


def _notary_summary(notary: dict[str, Any]) -> NotaryAccountSummary:
    return NotaryAccountSummary(
        notary_id=str(notary.get("_id", "")),
        user_id=notary.get("user_id", ""),
        full_name=notary.get("full_name", ""),
        registration_number=notary.get("registration_number", ""),
        jurisdiction_state=notary.get("jurisdiction_state", ""),
        verification_status=notary.get("verification_status", "unverified"),
        active=bool(notary.get("active", False)),
        certificate_reference=notary.get("certificate_reference", ""),
        verified_by_admin_id=notary.get("verified_by_admin_id"),
        verified_at=notary.get("verified_at"),
    )
