"""Pluggable e-signature provider layer.

No provider is hardcoded anywhere in this package. `registry.py` resolves a
provider by NAME from configuration, so integrating a real vendor is adding
one class and one registry entry -- no change to the service, API, or state
machine.
"""

from app.notarization.esign.base import (
    ESignProvider,
    SigningRequest,
    SigningSession,
    SigningStatus,
)
from app.notarization.esign.registry import available_providers, get_provider, register_provider

__all__ = [
    "ESignProvider",
    "SigningRequest",
    "SigningSession",
    "SigningStatus",
    "available_providers",
    "get_provider",
    "register_provider",
]
