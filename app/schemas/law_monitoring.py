"""Official-source monitoring inputs; checking a URL is never legal verification."""

import re
import ssl
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.kb_jurisdiction import normalize_state_code


def official_url(value: str) -> str:
    value = value.strip()
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment
            or not host.endswith((".gov.in", ".nic.in"))):
        raise ValueError("Use an HTTPS government .gov.in/.nic.in URL without credentials or fragment.")
    return value


# Confirmed live 2026-09-22: these servers' TLS stacks require a legacy
# renegotiation OpenSSL 3.x refuses by default (`[SSL:
# UNSAFE_LEGACY_RENEGOTIATION_DISABLED]`) -- affects the Bombay High Court's
# `bombay_high_court_judgments` catalogue adapter, the Bombay HC Maharashtra
# Acts library adapter, and the Payment of Wages Act (Maharashtra) official
# mirror, all on that host; and separately (same error, different host,
# confirmed independently) Chhattisgarh's `chhattisgarh_law_acts` adapter on
# `law.cgstate.gov.in`. A browser or `curl` on this machine connects fine to
# both (different TLS stack/defaults), so this is a server-side
# misconfiguration, not a client bug to route around everywhere.
#
# Scoped to exactly these hostnames, not a global relaxation: every OTHER
# fetch in this codebase keeps Python's ordinary, strict default TLS
# behaviour. Add a host here only after confirming (like these) that the
# SAME error reproduces against the real server, never speculatively.
_LEGACY_RENEGOTIATION_HOSTS = frozenset({
    "bombayhighcourt.gov.in", "bombayhighcourt.nic.in", "law.cgstate.gov.in",
})


@lru_cache(maxsize=1)
def _legacy_renegotiation_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    # Only present on OpenSSL 3.x builds (Python 3.12+); older builds allow
    # legacy renegotiation by default and need no flag at all.
    context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0)
    return context


# Confirmed live 2026-09-22: these two hosts' TLS 1.3 handshake hangs
# indefinitely against Python's OpenSSL client (`curl`/Windows Schannel on
# this same machine connects in well under a second -- a different TLS
# stack/negotiation path). Capping the client to TLS 1.2 makes the handshake
# complete immediately (confirmed directly against both hosts). A
# server-side TLS 1.3 misconfiguration, not a client bug to route around
# globally -- scoped to exactly these hostnames, the same pattern as the
# legacy-renegotiation case above, not a blanket downgrade.
_TLS12_ONLY_HOSTS = frozenset({"law.py.gov.in", "highcourtofuttarakhand.gov.in"})


@lru_cache(maxsize=1)
def _tls12_only_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def ssl_context_for_host(host: str | None) -> ssl.SSLContext | bool:
    """`verify=` value for `httpx.AsyncClient` -- a shared, scoped TLS
    workaround context for a known-affected host, `True` (httpx's own secure
    default) for every other host."""
    host_lower = (host or "").lower()
    if host_lower in _LEGACY_RENEGOTIATION_HOSTS:
        return _legacy_renegotiation_context()
    if host_lower in _TLS12_ONLY_HOSTS:
        return _tls12_only_context()
    return True


class MonitorRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    url: str = Field(max_length=2000)
    state_code: str | None = None
    topic: str = Field(min_length=1, max_length=100)
    interval_hours: int = Field(default=24, ge=1, le=168)
    enabled: bool = True
    content_selector: str | None = Field(default=None, min_length=1, max_length=200)

    _url = field_validator("url")(official_url)

    @field_validator("state_code")
    @classmethod
    def validate_state(cls, value: str | None) -> str | None:
        if value is None:
            return None
        code = normalize_state_code(value)
        if not code:
            raise ValueError("Unknown state/UT.")
        return code


class DocumentPublication(BaseModel):
    document_id: str = Field(min_length=1)
    jurisdiction_metadata: dict[str, Any]


class MonitorReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    decision: Literal["dismiss", "publish"]
    notes: str = Field(min_length=10, max_length=4000)
    evidence_url: str
    # Full metadata replacements, including explicit closures of affected old
    # versions. Dates/section overrides are human decisions, never inferred.
    publications: list[DocumentPublication] = Field(default_factory=list, max_length=20)
    affected_versions_reviewed: bool = False

    _url = field_validator("evidence_url")(official_url)


class CatalogueSourceRequest(BaseModel):
    """Admin-reviewed configuration for a State/UT or High Court catalogue."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    authority: str = Field(min_length=3, max_length=200)
    url: str = Field(max_length=2000)
    jurisdiction_code: str
    applicable_state_codes: list[str] = Field(default_factory=list, max_length=8)
    document_type: Literal[
        "bare_act", "rules", "regulation", "notification", "ordinance", "case_law",
        "circular", "order",
    ]
    include_pattern: str = Field(min_length=1, max_length=300)
    resolve_pdf_links: bool = False
    authority_confirmed: bool

    _url = field_validator("url")(official_url)

    @field_validator("jurisdiction_code")
    @classmethod
    def validate_jurisdiction(cls, value: str) -> str:
        code = normalize_state_code(value)
        if not code:
            raise ValueError("Unknown state/UT.")
        return code

    @field_validator("applicable_state_codes")
    @classmethod
    def validate_applicable_states(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            code = normalize_state_code(value)
            if not code:
                raise ValueError(f"Unknown state/UT: {value}")
            if code not in result:
                result.append(code)
        return result

    @field_validator("include_pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        try:
            re.compile(value, re.IGNORECASE)
        except re.error as exc:
            raise ValueError("Invalid catalogue include pattern.") from exc
        return value

    @model_validator(mode="after")
    def require_authority_confirmation(self) -> "CatalogueSourceRequest":
        if not self.authority_confirmed:
            raise ValueError("Confirm the URL belongs to the named issuing authority.")
        return self


class RelationshipReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    related_job_ids: list[str] = Field(min_length=1, max_length=20)
    notes: str = Field(min_length=10, max_length=4000)
