"""`streamlit_app/legal_source_review_page.py`'s pure logic: which staging
records surface as reviewable, and what payload a submission defaults to.

Loaded by path, same reason `test_chat_first_auth_and_kb_review.py` does --
`streamlit_app/app.py` would otherwise shadow the `app` package on import.
Nothing here renders Streamlit widgets; `_fetch_needs_review`/`_default_payload`
are plain functions over dicts and an HTTP stub.
"""

import importlib
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest


def _load() -> Any:
    # Unlike `auth_client.py` (deliberately dependency-free), this module does
    # a plain top-level `import auth_client` -- the same thing every other
    # `streamlit_app/*_page.py` module does, relying on Streamlit itself
    # putting the script's own directory on `sys.path`. Reproduced here rather
    # than changed there, so this test exercises the real import shape.
    #
    # Appended, never inserted at the front: `streamlit_app/app.py` would
    # otherwise shadow the real top-level `app` PACKAGE for the rest of the
    # test session the moment anything did `import app.anything` (confirmed
    # live: `conftest.py`'s own autouse fixture broke this way, since this
    # module is imported once at collection time and `sys.path` mutations
    # here outlive this file). Appending means the genuine `app` package
    # (found earlier on `sys.path`) always wins; `streamlit_app/` is only
    # consulted for names nothing else provides, e.g. `auth_client`.
    streamlit_app_dir = str(Path(__file__).resolve().parents[1] / "streamlit_app")
    if streamlit_app_dir not in sys.path:
        sys.path.append(streamlit_app_dir)
    return importlib.import_module("legal_source_review_page")


page = _load()


class _StubTransport(httpx.BaseTransport):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=self.payload)


def _client_factory(payload: dict[str, Any]):
    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_StubTransport(payload), base_url="http://testserver")

    return factory


_RECORDS = [
    {  # exactly the shape `/admin/knowledge-base/documents/needs-review` returns:
        # the server has already filtered to live needs_review documents.
        "document_id": "doc-needs-review",
        "filename": "IT_Amendment_Act_2008_MeitY_Scanned.pdf",
        "jurisdiction_metadata": {
            "document_key": "information-technology-amendment-act-2008", "issuing_level": "central",
            "applicability": "unknown", "jurisdiction_source_type": "bare_act",
            "verification_status": "unverified", "source_url": "https://www.meity.gov.in/x.pdf",
            "review_status": "needs_review", "review_reasons": ["Applicability is unknown."],
            "machine_verification": None, "jurisdiction_schema_version": 1,
            "metadata_provenance": "automated_official",
        },
    },
]


def test_fetch_calls_the_live_needs_review_endpoint_not_the_stale_staging_ledger(monkeypatch):
    """Regression: this page used to call the staging ledger's `status=all`
    listing and filter client-side. That ledger's `review_status` is a
    write-once snapshot from indexing time that never updates after a later
    approval -- confirmed live 2026-09-22 it reported 2,174 rows against
    only 21 real `needs_review` documents. The live endpoint returns an
    already-filtered, already-correct list -- no client-side filtering here."""
    session: dict[str, Any] = {}
    captured: dict[str, Any] = {}

    def fake_request(state, api_base_url, method, path, **kwargs):
        captured["path"] = path
        return httpx.Client(
            transport=_StubTransport(_RECORDS), base_url="http://testserver",
        ).request(method, path, **kwargs)

    monkeypatch.setattr(page.auth_client, "request", fake_request)
    records = page._fetch_needs_review(session, "http://testserver")
    assert captured["path"] == "/admin/knowledge-base/documents/needs-review"
    assert [record["document_id"] for record in records] == ["doc-needs-review"]


def test_fetch_returns_none_on_transport_error(monkeypatch):
    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(page.auth_client, "request", raise_error)
    assert page._fetch_needs_review({}, "http://testserver") is None


def test_default_payload_strips_server_managed_fields_and_fills_defaults():
    payload = page._default_payload(_RECORDS[0])
    assert "review_status" not in payload
    assert "review_reasons" not in payload
    assert "machine_verification" not in payload
    assert "jurisdiction_schema_version" not in payload
    assert "metadata_provenance" not in payload
    # Already-present fields survive untouched.
    assert payload["source_url"] == "https://www.meity.gov.in/x.pdf"
    assert payload["verification_status"] == "unverified"


def test_default_payload_strips_metadata_provenance_so_verified_is_not_silently_downgraded():
    """Regression: `normalize_jurisdiction` reads `metadata_provenance`
    straight off the submitted dict, in PREFERENCE to the server's own
    `provenance=PROVENANCE_MANUAL`. Leaving an automation-discovered
    document's `"automated_official"` value in here made every "Publish"
    with `verification_status: verified` silently become `inferred` instead
    -- confirmed live 2026-09-22 against real documents stuck exactly this
    way, never actually reaching `approved` no matter how many times the
    form was submitted."""
    record = {
        "document_id": "d1",
        "jurisdiction_metadata": {
            "issuing_level": "state", "applicability": "specific_states",
            "source_url": "https://law.jk.gov.in/act.pdf",
            "metadata_provenance": "automated_official",
        },
    }
    payload = page._default_payload(record)
    assert "metadata_provenance" not in payload


def test_default_payload_fills_defaults_for_a_document_with_no_jurisdiction_metadata_at_all():
    payload = page._default_payload({"document_id": "x"})
    assert payload["issuing_level"] == "central"
    assert payload["applicability"] == "unknown"
    assert payload["jurisdiction_source_type"] == "bare_act"
    assert payload["verification_status"] == "unverified"


@pytest.mark.parametrize("status", page._VERIFICATION_STATUSES)
def test_machine_verified_is_never_an_offerable_choice(status):
    assert status != "machine_verified"


# ------------------------------------------------------------- fast-review signals


_STRONG_PAYLOAD = {
    "issuing_level": "central", "applicability": "all_india", "jurisdiction_source_type": "bare_act",
    "source_url": "https://www.meity.gov.in/static/uploads/x.pdf",
    "machine_verification": {"identity_checks": [True, True]},
}


def test_all_signals_pass_for_a_fully_evidenced_official_record():
    signals = page._signals(_STRONG_PAYLOAD)
    assert all(passed for passed, _ in signals)


def test_non_government_url_fails_the_official_domain_signal():
    payload = {**_STRONG_PAYLOAD, "source_url": "https://example.com/act.pdf"}
    passed, label = page._signals(payload)[0]
    assert not passed and "gov.in" in label


@pytest.mark.parametrize("missing_url", ["", None])
def test_missing_url_fails_the_official_domain_signal(missing_url):
    payload = {**_STRONG_PAYLOAD, "source_url": missing_url}
    passed, _ = page._signals(payload)[0]
    assert not passed


@pytest.mark.parametrize("evidence", [None, {}, {"identity_checks": []}, {"identity_checks": [False]}, "not-a-dict"])
def test_absent_or_failed_identity_evidence_fails_that_signal(evidence):
    payload = {**_STRONG_PAYLOAD, "machine_verification": evidence}
    passed, _ = page._signals(payload)[1]
    assert not passed


def test_automation_discovery_evidence_shape_also_passes_the_middle_signal():
    """`kb_automation.py::_process_claimed` (the state/central adapter
    discovery path) never writes `identity_checks` -- only `download_sha256`,
    proving the exact file on disk was fetched from `source_url` at
    discovery time. Without this, the middle signal -- and therefore every
    "all green" shortcut, single or bulk -- was unreachable for every
    automation-discovered document."""
    payload = {
        **_STRONG_PAYLOAD,
        "machine_verification": {"download_sha256": "abc123", "downloaded_at": "2026-09-22T00:00:00Z"},
    }
    passed, label = page._signals(payload)[1]
    assert passed
    assert "downloaded and hashed" in label.lower()


def test_signals_for_record_sees_machine_verification_that_default_payload_strips():
    """Regression: `_default_payload` deliberately pops `machine_verification`
    (it must never be resubmitted to the server as a stale copy on Publish) --
    but `_review_card`/`render` used to call `_signals(_default_payload(record))`
    directly, so the evidence signal was ALWAYS unreachable, for every
    document, regardless of what real evidence it had. Confirmed live
    2026-09-22 against a real automation-discovered document that genuinely
    has `download_sha256`: signal 2 still showed failed until this fix.
    `_signals_for_record` re-injects the raw (never-resubmitted) evidence."""
    record = {
        "document_id": "d1",
        "jurisdiction_metadata": {
            "issuing_level": "state", "applicability": "specific_states",
            "jurisdiction_source_type": "bare_act", "source_url": "https://law.jk.gov.in/act.pdf",
            "machine_verification": {"download_sha256": "abc123"},
        },
    }
    payload = page._default_payload(record)
    assert payload.get("machine_verification") is None  # confirms the strip still happens
    signals = page._signals_for_record(record, payload)
    assert all(passed for passed, _ in signals)


def test_automation_discovery_evidence_alone_makes_every_signal_pass():
    payload = {
        "issuing_level": "state", "applicability": "specific_states", "jurisdiction_source_type": "bare_act",
        "source_url": "https://law.jk.gov.in/act.pdf",
        "machine_verification": {"download_sha256": "abc123"},
    }
    assert all(passed for passed, _ in page._signals(payload))


@pytest.mark.parametrize("field", ["issuing_level", "applicability", "jurisdiction_source_type"])
def test_an_unknown_field_fails_the_completeness_signal(field):
    payload = {**_STRONG_PAYLOAD, field: "unknown"}
    passed, _ = page._signals(payload)[2]
    assert not passed


def test_quick_verify_payload_sets_verified_and_keeps_every_other_field():
    body = page._quick_verify_payload(_STRONG_PAYLOAD, "reviewer@example.com (checked against MeitY PDF)")
    assert body["verification_status"] == "verified"
    assert body["verified_by"] == "reviewer@example.com (checked against MeitY PDF)"
    for key, value in _STRONG_PAYLOAD.items():
        assert body[key] == value


def test_quick_verify_payload_never_defaults_a_blank_reviewer_identity():
    body = page._quick_verify_payload(_STRONG_PAYLOAD, "   ")
    assert body["verified_by"] is None


# ------------------------------------------------------------- bulk approve


class _CapturingTransport(httpx.BaseTransport):
    """Records the request body sent, then answers with a fixed payload --
    lets a test assert both what was SENT (the right document_ids/verified_by)
    and what the page does with what comes back."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return httpx.Response(200, json=self.payload)


def _patch_request(monkeypatch: pytest.MonkeyPatch, transport: httpx.BaseTransport) -> None:
    monkeypatch.setattr(
        page.auth_client, "request",
        lambda state, api_base_url, method, path, **kwargs: httpx.Client(
            transport=transport, base_url="http://testserver",
        ).request(method, path, **kwargs),
    )


def test_bulk_submit_sends_every_selected_id_and_the_reviewer_identity(monkeypatch):
    transport = _CapturingTransport({"results": [
        {"document_id": "d1", "outcome": "approved"},
        {"document_id": "d2", "outcome": "approved"},
    ]})
    _patch_request(monkeypatch, transport)
    monkeypatch.setattr(page.st, "toast", lambda *a, **k: None)
    monkeypatch.setattr(page.st, "rerun", lambda: None)

    page._bulk_submit({}, "http://testserver", ["d1", "d2"], "reviewer@example.com")

    assert transport.last_request is not None
    assert transport.last_request.url.path == "/admin/knowledge-base/documents/bulk-review"
    import json as _json
    sent = _json.loads(transport.last_request.content)
    assert sent == {"document_ids": ["d1", "d2"], "verified_by": "reviewer@example.com"}


def test_bulk_submit_reports_documents_that_stayed_needs_review(monkeypatch):
    """The core safety guarantee: a document that fails the server-side check
    is reported back as still-pending, not silently swallowed as a success."""
    transport = _CapturingTransport({"results": [
        {"document_id": "d1", "outcome": "approved"},
        {"document_id": "d2", "outcome": "needs_review", "review_reasons": ["Issuing level is unknown."]},
    ]})
    _patch_request(monkeypatch, transport)
    warnings: list[str] = []
    monkeypatch.setattr(page.st, "toast", lambda *a, **k: None)
    monkeypatch.setattr(page.st, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(page.st, "rerun", lambda: None)

    page._bulk_submit({}, "http://testserver", ["d1", "d2"], "reviewer@example.com")

    assert any("1 document(s) stayed" in msg for msg in warnings)


def test_bulk_submit_reports_a_failed_document_without_losing_the_others(monkeypatch):
    transport = _CapturingTransport({"results": [
        {"document_id": "d1", "outcome": "approved"},
        {"document_id": "missing", "outcome": "failed", "reason": "Document not found."},
    ]})
    _patch_request(monkeypatch, transport)
    errors: list[str] = []
    monkeypatch.setattr(page.st, "toast", lambda *a, **k: None)
    monkeypatch.setattr(page.st, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(page.st, "rerun", lambda: None)

    page._bulk_submit({}, "http://testserver", ["d1", "missing"], "reviewer@example.com")

    assert any("Document not found." in msg for msg in errors)


def test_bulk_submit_shows_an_error_on_transport_failure(monkeypatch):
    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(page.auth_client, "request", raise_error)
    errors: list[str] = []
    monkeypatch.setattr(page.st, "error", lambda msg: errors.append(msg))

    page._bulk_submit({}, "http://testserver", ["d1"], "reviewer@example.com")

    assert errors
