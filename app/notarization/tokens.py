"""Signed, expiring action tokens for notarization flows.

Used for the two places where an action must be authorised out-of-band and
must not be replayable indefinitely:

* an e-sign provider callback, which arrives from outside our trust boundary
  and must prove it corresponds to a signing session WE started;
* a notary's approve/reject/revoke step, which requires a fresh
  re-authentication rather than merely a still-valid login session.

Deliberately NOT a JWT: these are internal, short-lived, single-purpose
capabilities with no need for a claims ecosystem, and hand-rolling the
minimum (HMAC-SHA256 over a compact payload, constant-time compare, explicit
expiry) keeps the security surface small and readable. The signing key is the
app's existing `jwt_secret_key`, whose production strength `Settings.
_validate_security` already enforces.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import ForbiddenError


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _signature(payload: str) -> str:
    return _b64(hmac.new(settings.jwt_secret_key.encode(), payload.encode(), hashlib.sha256).digest())


def issue_action_token(purpose: str, subject: str, ttl_seconds: int, **claims: Any) -> str:
    """A signed capability to perform `purpose` on `subject`, valid for `ttl_seconds`."""
    body = {"purpose": purpose, "subject": subject, "exp": int(time.time()) + ttl_seconds, **claims}
    payload = _b64(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    return f"{payload}.{_signature(payload)}"


def verify_action_token(token: str, purpose: str, subject: str) -> dict[str, Any]:
    """Validates `token` and returns its claims, or raises `ForbiddenError`.

    Checks, in order: structure, signature, expiry, purpose, subject. Purpose
    and subject are both checked so a token issued to approve request A can
    never be replayed to approve request B, nor an approve token reused to
    revoke.
    """
    payload, _, provided_signature = (token or "").partition(".")
    if not payload or not provided_signature:
        raise ForbiddenError("Malformed action token.")
    if not hmac.compare_digest(provided_signature, _signature(payload)):
        raise ForbiddenError("Action token signature is invalid.")
    try:
        claims = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ForbiddenError("Malformed action token.") from exc
    if claims.get("exp", 0) < time.time():
        raise ForbiddenError("This action token has expired. Please re-authenticate and try again.")
    if claims.get("purpose") != purpose:
        raise ForbiddenError("This action token was not issued for this operation.")
    if claims.get("subject") != subject:
        raise ForbiddenError("This action token was not issued for this document.")
    if not isinstance(claims, dict):
        raise ForbiddenError("Malformed action token.")
    return claims
