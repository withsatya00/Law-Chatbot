"""Request/response models for the E-Notarization API.

Note what the public verification response deliberately omits: document
content, signer address, contact details, identity numbers, and the owning
user's identity. See `PublicVerificationResponse`.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.notarization.esign.base import SigningStatus
from app.notarization.states import DocumentStatus

# Fixed warnings that must appear wherever a document's status is shown. Kept
# server-side so every client shows the same words -- a client that composes
# its own could soften them.
AI_DRAFT_WARNING = "This is an AI-generated draft, not a notarized document."
ESIGN_WARNING = "Digital signing alone may not constitute notarization."
VALIDITY_WARNING = (
    "Final legal validity depends on applicable law and verification by a licensed notary."
)
STANDARD_WARNINGS: tuple[str, ...] = (AI_DRAFT_WARNING, ESIGN_WARNING, VALIDITY_WARNING)


class SignerDetails(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    address: str = Field(min_length=1, max_length=500)
    # TYPE only. The number is never collected: the notary inspects the
    # original document in person.
    identity_document_type: Literal["Aadhaar", "PAN", "Passport", "VoterID", "DrivingLicence", "Other"]
    place: str = Field(min_length=1, max_length=200)
    date: str = Field(min_length=1, max_length=40)
    email: str = Field(default="", max_length=320)


class WitnessDetails(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    address: str = Field(min_length=1, max_length=500)


class ChecklistItemModel(BaseModel):
    key: str
    label: str
    severity: Literal["required", "recommended"]
    satisfied: bool
    detail: str = ""


class PrepareNotarizationRequest(BaseModel):
    draft_id: str = Field(min_length=1)
    signer: SignerDetails
    witnesses: list[WitnessDetails] = Field(default_factory=list)


class PrepareNotarizationResponse(BaseModel):
    document_id: str
    document_version: int
    document_hash: str
    status: DocumentStatus
    eligible: bool
    checklist_complete: bool
    checklist: list[ChecklistItemModel] = Field(default_factory=list)
    # "Draft" or "Notary-ready draft". Never "Notarized".
    label: str
    ineligible_reason: str = ""
    warnings: list[str] = Field(default_factory=lambda: list(STANDARD_WARNINGS))


class InitiateSigningRequest(BaseModel):
    document_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1, max_length=300)
    provider: str | None = None


class SigningConsent(BaseModel):
    """Exactly what the user is shown before any signature is applied."""

    document_title: str
    document_hash: str
    signer_name: str
    purpose: str
    signing_timestamp: datetime
    warnings: list[str] = Field(default_factory=lambda: list(STANDARD_WARNINGS))


class InitiateSigningResponse(BaseModel):
    session_id: str
    document_id: str
    provider: str
    status: SigningStatus
    signing_url: str = ""
    consent: SigningConsent
    message: str = ""
    # Security finding N1: previously computed by `NotarizationService.
    # initiate_signing` and then silently dropped by this response, so
    # nothing -- not the client, not the configured signing provider --
    # ever actually received it, making `POST /notarization/signing/
    # callback`'s `verify_action_token` check impossible to ever satisfy
    # legitimately. Also now embedded directly in the callback URL handed
    # to the provider itself (`SigningRequest.callback_url`); returned here
    # too for a provider/flow that instead expects the caller's own
    # frontend to relay it.
    callback_token: str = ""


class SigningCallbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    action_token: str = Field(min_length=1)
    status: SigningStatus
    provider_reference: str = ""
    failure_reason: str = Field(default="", max_length=500)


class CreateNotarizationRequestPayload(BaseModel):
    document_id: str = Field(min_length=1)
    assigned_notary_id: str | None = None
    supporting_id_confirmed: bool = False
    note: str = Field(default="", max_length=1000)


class NotarizationRequestSummary(BaseModel):
    request_id: str
    document_id: str
    document_version: int
    document_title: str
    document_hash: str
    review_status: Literal["pending", "in_review", "approved", "rejected"]
    signer_name: str
    supporting_id_confirmed: bool
    witness_count: int
    assigned_notary_id: str | None = None
    remarks: str = ""
    created_at: datetime | None = None
    reviewed_at: datetime | None = None


class NotaryDecisionRequest(BaseModel):
    """Approve/reject/revoke. `action_token` is the fresh re-authentication."""

    action_token: str = Field(min_length=1)
    remarks: str = Field(default="", max_length=1000)
    # Recomputed by the notary's client from the document they actually
    # reviewed. The server compares it to the stored hash and refuses on
    # mismatch, so a notary can never approve bytes they did not see.
    confirmed_document_hash: str = Field(default="", max_length=64)


class NotaryDecisionResponse(BaseModel):
    request_id: str
    document_id: str
    review_status: str
    document_status: DocumentStatus
    verification_url: str = ""
    message: str = ""


class NotaryAccountPayload(BaseModel):
    user_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1, max_length=200)
    registration_number: str = Field(min_length=1, max_length=100)
    jurisdiction_state: str = Field(min_length=1, max_length=100)
    certificate_reference: str = Field(default="", max_length=200)


class NotaryAccountSummary(BaseModel):
    notary_id: str
    user_id: str
    full_name: str
    registration_number: str
    jurisdiction_state: str
    verification_status: Literal["unverified", "verified", "revoked"]
    active: bool
    certificate_reference: str = ""
    verified_by_admin_id: str | None = None
    verified_at: datetime | None = None


class DocumentStatusTimelineEntry(BaseModel):
    action: str
    actor_role: str
    document_version: int
    occurred_at: datetime
    detail: dict[str, Any] = Field(default_factory=dict[str, Any])


class DocumentStatusResponse(BaseModel):
    document_id: str
    document_version: int
    document_title: str
    status: DocumentStatus
    status_label: str
    document_hash: str
    is_notarized: bool
    downloadable: bool
    timeline: list[DocumentStatusTimelineEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=lambda: list(STANDARD_WARNINGS))


class PublicVerificationResponse(BaseModel):
    """The ONLY thing the public verification endpoint returns.

    Every field here is deliberate, and everything absent is deliberate too:
    no document content, no signer address or contact details, no identity
    document information, no owning-user identity, and no internal ids. A
    verifier needs to know that a document is genuine and who attested it --
    not who the parties are or what the document says.
    """

    found: bool
    status: str = ""
    status_label: str = ""
    document_type: str = ""
    document_title: str = ""
    hash_matches: bool | None = None
    notarized_at: datetime | None = None
    notary_name: str = ""
    notary_registration_number: str = ""
    revoked: bool = False
    revoked_at: datetime | None = None
    message: str = ""
