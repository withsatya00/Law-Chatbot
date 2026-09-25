"""Immutable notarization audit events.

"Immutable" here is enforced at the only layer that can actually enforce it
in this codebase: `NotarizationAuditRepository` exposes append and read, and
deliberately does NOT inherit the base repository's `update_by_id` /
`delete_by_id`. There is no code path in the application that can alter or
remove a recorded event. (Database-level protection -- a append-only role, or
WORM storage -- is an operator concern documented in docs/NOTARIZATION.md;
this closes the application-level hole.)

Every event carries, without exception: when it happened, WHO did it and in
what role, which document and which VERSION of it, and what the action was.
The document version matters most: an audit trail that says "approved" without
saying which bytes were approved cannot establish anything later.
"""

from datetime import UTC, datetime
from typing import Any, Literal

AuditAction = Literal[
    "draft_generated",
    "user_edited",
    "document_exported",
    "notary_ready_prepared",
    "esign_initiated",
    "esign_completed",
    "esign_failed",
    "notarization_requested",
    "notary_review_started",
    "notary_approved",
    "notary_rejected",
    "notarized_version_issued",
    "notarization_revoked",
    "notary_account_verified",
    "notary_account_revoked",
    "verification_viewed",
]

ActorRole = Literal["user", "notary", "admin", "system", "provider"]


def build_event(
    *,
    action: AuditAction,
    document_id: str,
    document_version: int,
    actor_id: str | None,
    actor_role: ActorRole,
    document_hash: str = "",
    request_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One audit record.

    `detail` is for non-sensitive context only (a rejection reason, a
    provider name, a status transition). Never place OTPs, identity images,
    identity numbers, secret keys, or document content in it -- see
    `redact_detail`, which is applied on the way in.
    """
    return {
        "action": action,
        "document_id": document_id,
        "document_version": document_version,
        "document_hash": document_hash,
        "actor_id": actor_id,
        "actor_role": actor_role,
        "request_id": request_id,
        "detail": redact_detail(detail or {}),
        "occurred_at": datetime.now(UTC),
    }


# Keys that must never reach storage or logs, whatever a caller passes.
# Matched as substrings, case-insensitively, so "aadhaar_number",
# "signer_otp" and "provider_api_key" are all caught.
_FORBIDDEN_DETAIL_KEYS = (
    "aadhaar", "aadhar", "otp", "biometric", "fingerprint", "iris",
    "password", "secret", "api_key", "apikey", "token", "credential",
    "identity_number", "id_number", "document_content", "content",
)

_REDACTED = "[redacted]"


def redact_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Strips forbidden keys from an audit detail payload.

    A guard, not a substitute for callers being careful: it exists so that a
    future caller who passes something sensitive by accident cannot persist
    it. Redacted rather than dropped so the attempt itself stays visible.
    """
    cleaned: dict[str, Any] = {}
    for key, value in (detail or {}).items():
        lowered = str(key).lower()
        if any(forbidden in lowered for forbidden in _FORBIDDEN_DETAIL_KEYS):
            cleaned[key] = _REDACTED
            continue
        cleaned[key] = redact_detail(value) if isinstance(value, dict) else value
    return cleaned
