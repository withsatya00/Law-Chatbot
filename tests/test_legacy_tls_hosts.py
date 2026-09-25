"""`ssl_context_for_host`: scoped TLS workarounds for servers confirmed to
need them -- `[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED]` against
`bombayhighcourt.gov.in`/`.nic.in` and (confirmed independently, same error)
`law.cgstate.gov.in` (2026-09-22), and a TLS 1.3 handshake hang against
`law.py.gov.in`/`highcourtofuttarakhand.gov.in` (2026-09-22, fixed by capping
the client to TLS 1.2) -- never a blanket weakening for every fetch this
project makes.
"""

import ssl

import pytest

from app.schemas.law_monitoring import ssl_context_for_host


@pytest.mark.parametrize("host", ["bombayhighcourt.gov.in", "law.cgstate.gov.in"])
def test_known_affected_host_gets_a_legacy_renegotiation_context(host):
    result = ssl_context_for_host(host)
    assert isinstance(result, ssl.SSLContext)


def test_matching_is_case_insensitive():
    assert isinstance(ssl_context_for_host("BombayHighCourt.Gov.In"), ssl.SSLContext)


@pytest.mark.parametrize("host", ["law.py.gov.in", "highcourtofuttarakhand.gov.in"])
def test_tls13_hang_host_gets_a_tls12_capped_context(host):
    result = ssl_context_for_host(host)
    assert isinstance(result, ssl.SSLContext)
    assert result.maximum_version == ssl.TLSVersion.TLSv1_2


def test_tls13_hang_matching_is_case_insensitive():
    result = ssl_context_for_host("Law.PY.Gov.In")
    assert isinstance(result, ssl.SSLContext)
    assert result.maximum_version == ssl.TLSVersion.TLSv1_2


@pytest.mark.parametrize(
    "host",
    ["indiacode.nic.in", "mha.gov.in", "example.com", "notbombayhighcourt.gov.in", ""],
)
def test_every_other_host_keeps_the_ordinary_secure_default(host):
    assert ssl_context_for_host(host) is True


def test_none_host_keeps_the_ordinary_secure_default():
    assert ssl_context_for_host(None) is True


def test_context_is_reused_not_rebuilt_per_call():
    # `@lru_cache` on the underlying factory -- constructing an SSLContext is
    # not free, and this is looked up on every fetch to a known-affected host.
    first = ssl_context_for_host("bombayhighcourt.gov.in")
    second = ssl_context_for_host("bombayhighcourt.nic.in")
    assert first is second

    first_tls12 = ssl_context_for_host("law.py.gov.in")
    second_tls12 = ssl_context_for_host("highcourtofuttarakhand.gov.in")
    assert first_tls12 is second_tls12
