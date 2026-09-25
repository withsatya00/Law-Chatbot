"""The default provider: no external e-sign vendor configured.

This is NOT a simulated signature. It never reports "signed" on its own, and
it has no code path that can. It exists so the platform has honest behaviour
out of the box: a deployment with no vendor integration gets a signing
session that stays `pending` and says so, instead of either crashing or --
far worse -- pretending a signature exists.

Advancing a document to `signed` through this provider requires an
out-of-band, human-performed wet or digital signature to be recorded by an
operator, which is a deliberate, auditable act rather than something this
class can do by itself.
"""

from datetime import UTC, datetime

from app.notarization.esign.base import ESignProvider, SigningRequest, SigningSession


class ManualESignProvider(ESignProvider):
    name = "manual"

    _UNCONFIGURED = (
        "No e-signature provider is configured for this deployment. The document has been prepared for "
        "signature, but no electronic signature has been applied to it."
    )

    async def initiate(self, request: SigningRequest) -> SigningSession:
        return SigningSession(
            provider=self.name,
            provider_reference=f"manual:{request.document_id}:{request.document_version}",
            status="pending",
            failure_reason=self._UNCONFIGURED,
            metadata={"document_hash": request.document_hash, "requested_at": datetime.now(UTC).isoformat()},
        )

    async def fetch_status(self, provider_reference: str) -> SigningSession:
        return SigningSession(
            provider=self.name,
            provider_reference=provider_reference,
            status="pending",
            failure_reason=self._UNCONFIGURED,
        )

    async def cancel(self, provider_reference: str) -> SigningSession:
        return SigningSession(provider=self.name, provider_reference=provider_reference, status="cancelled")
