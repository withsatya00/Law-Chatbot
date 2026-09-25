"""The e-sign provider contract.

Any vendor integration implements `ESignProvider`. The rest of the system
knows only this interface and the six `SigningStatus` values, so swapping or
adding a provider never touches the notarization state machine.

Privacy constraint enforced by the shape of these types, not merely by
convention: `SigningSession` carries no Aadhaar number, no OTP, no biometric
data, and no provider credential. There is nowhere in this contract to put
them, so an integration cannot casually persist them through this path. What
IS carried is the minimum needed to correlate a signature with a document:
an opaque provider reference, a status, a signer display name/email, and
timestamps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# The six states every provider is normalised onto, regardless of the vendor's
# own vocabulary. `pending` is "we have created our record but not yet called
# the provider"; `initiated` is "the provider has a signing ceremony open".
SigningStatus = Literal["pending", "initiated", "signed", "failed", "expired", "cancelled"]

TERMINAL_SIGNING_STATUSES: frozenset[str] = frozenset({"signed", "failed", "expired", "cancelled"})
# Outcomes that must leave the document NON-signed.
UNSUCCESSFUL_SIGNING_STATUSES: frozenset[str] = frozenset({"failed", "expired", "cancelled"})


@dataclass(frozen=True)
class SigningRequest:
    """What the platform asks a provider to do.

    `document_hash` rather than the document itself: the provider is told
    exactly which bytes are being signed, and our record of the signature is
    bound to that hash, so a later content change is detectable.
    """

    document_id: str
    document_version: int
    document_title: str
    document_hash: str
    signer_name: str
    signer_email: str
    purpose: str
    callback_url: str = ""


@dataclass(frozen=True)
class SigningSession:
    """A provider's response, normalised.

    `provider_reference` is whatever opaque id the vendor uses; it is stored
    so a callback can be correlated, and is never treated as proof of
    anything on its own.
    """

    provider: str
    provider_reference: str
    status: SigningStatus
    signing_url: str = ""
    signed_at: datetime | None = None
    failure_reason: str = ""
    # Non-sensitive provider metadata only. Documented here because it is the
    # one open-ended field: integrations MUST NOT place identity numbers,
    # OTPs, biometric payloads or credentials in it.
    metadata: dict[str, str] = field(default_factory=dict)


class ESignProvider(ABC):
    """Interface every e-sign integration implements."""

    name: str

    @abstractmethod
    async def initiate(self, request: SigningRequest) -> SigningSession:
        """Opens a signing ceremony. Must not raise for ordinary vendor
        failures -- return a `SigningSession` with status "failed" and a
        `failure_reason` instead, so the caller can record the outcome."""

    @abstractmethod
    async def fetch_status(self, provider_reference: str) -> SigningSession:
        """Current status, for polling or for validating a callback."""

    @abstractmethod
    async def cancel(self, provider_reference: str) -> SigningSession:
        """Cancels an open ceremony."""
