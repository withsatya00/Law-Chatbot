"""Name -> provider resolution.

The indirection is the point: `ESIGN_PROVIDER` names a provider, and nothing
in the service, API or state machine mentions a vendor. Adding DocuSign,
Digio, Leegality, eMudhra or an internal service is a new module plus one
`register_provider` call.
"""

import structlog

from app.core.config import settings
from app.core.exceptions import BadRequestError
from app.notarization.esign.base import ESignProvider
from app.notarization.esign.manual import ManualESignProvider

log = structlog.get_logger(__name__)

_PROVIDERS: dict[str, type[ESignProvider]] = {}


def register_provider(provider_class: type[ESignProvider]) -> type[ESignProvider]:
    """Registers a provider class under its own `name`. Usable as a decorator."""
    name = getattr(provider_class, "name", "")
    if not name:
        raise ValueError("An e-sign provider must define a non-empty `name`.")
    _PROVIDERS[name.lower()] = provider_class
    return provider_class


register_provider(ManualESignProvider)


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str | None = None) -> ESignProvider:
    """The configured provider, or the one explicitly named.

    An unknown name is an error rather than a silent fallback to `manual`:
    a deployment that believes it configured a real vendor must not quietly
    end up issuing pending-forever sessions instead.
    """
    resolved = (name or settings.esign_provider or "manual").strip().lower()
    provider_class = _PROVIDERS.get(resolved)
    if provider_class is None:
        raise BadRequestError(
            f"Unknown e-sign provider '{resolved}'. Registered providers: {', '.join(available_providers())}.",
            {"provider": resolved, "available": available_providers()},
        )
    return provider_class()
