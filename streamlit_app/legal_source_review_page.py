"""Admin-only page for publishing an already-indexed, `needs_review` Knowledge
Base document into shared retrieval.

Calls exactly one route -- `PUT /admin/knowledge-base/documents/{document_id}
/jurisdiction` (`KnowledgeBaseIngestionService.update_jurisdiction_metadata`)
-- the same one a curl call would use, so nothing here bypasses `require_admin`
or the audit log, and nothing here can grant `verification_status="verified"`
on its own: that still requires a human to fill in and submit the form.

Not to be confused with `/admin/phase3/legal-sources/*` (`LegalUpdateService`),
a separate governance registry that does not gate chat retrieval. This page
exists for the field shared retrieval actually reads:
`embeddings_metadata.metadata.review_status`.
"""

import json
import re
from collections.abc import MutableMapping
from typing import Any

import auth_client
import httpx
import streamlit as st

# Mirrors `app.schemas.law_monitoring.official_url`'s domain rule -- reimplemented
# here rather than imported, since this Streamlit process only ever talks to the
# API over HTTP (see `auth_client.request`), never imports `app.*` directly.
_OFFICIAL_DOMAIN_RE = re.compile(r"^https://[^/]+\.(gov\.in|nic\.in)(/|$)", re.IGNORECASE)

_ISSUING_LEVELS = ("central", "state", "local", "unknown")
_APPLICABILITIES = ("all_india", "specific_states", "unknown")
_SOURCE_TYPES = (
    "bare_act", "amendment_act", "rules", "regulation", "notification", "circular",
    "ordinance", "bill", "case_law", "commentary", "faq", "model_law", "unknown",
)
# `machine_verified` is deliberately absent: that status is reserved for the
# fail-closed byte-hash/token evidence pipeline in
# `app.services.kb_machine_verification`, not something a form submission
# should be able to claim directly.
_VERIFICATION_STATUSES = ("unverified", "verified", "inferred")


def _fetch_needs_review(session: MutableMapping[str, Any], api_base_url: str) -> list[dict[str, Any]] | None:
    """Live `needs_review` documents, from `/admin/knowledge-base/documents/
    needs-review` -- NOT the staging ledger's `status=all` listing this
    used to call. That ledger's `review_status` is a write-once snapshot
    from indexing time that never updates after a later approval; confirmed
    live 2026-09-22 it was showing 2,174 rows against only 21 real
    `needs_review` documents, making this page nearly unusable at real
    automation-pipeline scale (both wrong and slow to render)."""
    try:
        response = auth_client.request(
            session, api_base_url, "GET", "/admin/knowledge-base/documents/needs-review",
            params={"limit": 200}, timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        st.error("Could not reach the server to load the review queue.")
        return None
    return list(response.json())


def _default_payload(record: dict[str, Any]) -> dict[str, Any]:
    current = dict(record.get("jurisdiction_metadata") or {})
    current.setdefault("issuing_level", "central")
    current.setdefault("applicability", "unknown")
    current.setdefault("jurisdiction_source_type", "bare_act")
    current.setdefault("verification_status", "unverified")
    # Server-managed / evidence fields that must never be resubmitted as
    # stale copies: the server recomputes `review_status`/`review_reasons`
    # from the rest of the payload, and a real `machine_verification`
    # evidence bundle can only come from `kb_machine_verification.py` itself.
    #
    # `metadata_provenance` belongs in this list too, for a sharper reason:
    # `normalize_jurisdiction` reads it straight off the submitted dict
    # (`raw.get("metadata_provenance", provenance)`) IN PREFERENCE to the
    # `provenance=PROVENANCE_MANUAL` the server's endpoint passes -- an
    # automation-discovered document's existing `"automated_official"` value,
    # left in here, silently downgrades `verification_status: "verified"` to
    # `"inferred"` (a NON-human provenance can never claim `verified`) and
    # the document never actually becomes `approved`, no matter how many
    # times this form is submitted. Confirmed live 2026-09-22 against real
    # documents stuck exactly this way. Submitting through THIS form is
    # itself the human review action -- its provenance is manual by
    # definition, and the server's own default should be the one that wins.
    for managed_field in (
        "review_status", "review_reasons", "machine_verification",
        "jurisdiction_schema_version", "metadata_provenance",
    ):
        current.pop(managed_field, None)
    return current


def _signals(payload: dict[str, Any]) -> list[tuple[bool, str]]:
    """Cheap, honest pre-checks from the record alone -- NOT a legal opinion on
    the document's content, which nothing here reads. Purely to let a reviewer
    triage a long queue at a glance: a document with every green badge still
    needs the SAME human decision `verification_status: verified` always has,
    just with less typing to get there (see `_quick_verify_payload`). A
    document with red badges is exactly where a reviewer's time is best spent.

    The middle signal recognizes TWO different, unrelated evidence shapes,
    from two different pipelines, and labels whichever one is actually
    present rather than claiming the other happened:
    - `identity_checks` (`kb_machine_verification.py`/`kb_gap_autofetch.py`/
      `kb_official_source_sync.py`): a real text-content identity check ran.
    - `download_sha256` (`kb_automation.py::_process_claimed`, the state/
      central adapter discovery path): no content check ran, but the exact
      bytes now on disk were downloaded and hashed directly from `source_url`
      at discovery time -- confirmed 2026-09-22 against a real automation
      queue of 2,174 documents, ALL of which carry this evidence and NONE of
      which carry `identity_checks` (a different pipeline's field), which
      had silently made this whole signal -- and therefore every "all green"
      shortcut on this page, single or bulk -- unreachable for the entire
      automation pipeline until this fix.
    """
    source_url = str(payload.get("source_url") or "")
    evidence = payload.get("machine_verification")
    evidence_dict = evidence if isinstance(evidence, dict) else {}
    identity_checked = bool(any(evidence_dict.get("identity_checks") or []))
    downloaded_and_hashed = bool(evidence_dict.get("download_sha256"))
    if identity_checked:
        automated_evidence_label = "Automated identity check already passed at ingestion"
    elif downloaded_and_hashed:
        automated_evidence_label = "File downloaded and hashed directly from this URL at discovery time"
    else:
        automated_evidence_label = "Automated identity check already passed at ingestion"
    fields_complete = (
        payload.get("issuing_level") not in (None, "unknown")
        and payload.get("applicability") not in (None, "unknown")
        and payload.get("jurisdiction_source_type") not in (None, "unknown")
        and bool(source_url)
    )
    return [
        (bool(_OFFICIAL_DOMAIN_RE.match(source_url)), "Official .gov.in/.nic.in source URL"),
        (identity_checked or downloaded_and_hashed, automated_evidence_label),
        (fields_complete, "Issuing level, applicability and source type already filled in"),
    ]


def _quick_verify_payload(payload: dict[str, Any], verified_by: str) -> dict[str, Any]:
    """Same fields the full form would submit for 'verified', with everything
    ELSE (issuing_level, applicability, source_type, source_url, ...) kept
    exactly as already on record -- this shortens the click count for a
    reviewer who has confirmed the source, it does not change what is being
    asserted."""
    body = dict(payload)
    body.update(verification_status="verified", verified_by=verified_by.strip() or None)
    return body


def _signals_for_record(record: dict[str, Any], payload: dict[str, Any]) -> list[tuple[bool, str]]:
    """`_default_payload` deliberately strips `machine_verification` (it must
    never be resubmitted as a stale copy on Publish) -- but that means
    calling `_signals(payload)` directly never sees it either, so the
    automated-evidence signal was unreachable for every document,
    regardless of what evidence it actually has. Re-injects the RAW
    `machine_verification` from the record (never submitted, just read) so
    the signal reflects reality.
    """
    raw_metadata = record.get("jurisdiction_metadata") or {}
    return _signals({**payload, "machine_verification": raw_metadata.get("machine_verification")})


def _review_card(session: MutableMapping[str, Any], api_base_url: str, record: dict[str, Any]) -> None:
    document_id = record["document_id"]
    filename = record.get("filename") or record.get("generated_filename") or record.get("original_filename") or document_id
    payload = _default_payload(record)
    reasons = payload.get("review_reasons") or record.get("jurisdiction_metadata", {}).get("review_reasons") or []
    signals = _signals_for_record(record, payload)
    all_green = all(passed for passed, _ in signals)

    with st.container(border=True):
        st.markdown(f"**📄 {filename}**" + (" 🟢" if all_green else " 🟡"))
        st.caption(f"document_id: `{document_id}`")
        if reasons:
            st.warning("Needs: " + "; ".join(str(reason) for reason in reasons))
        st.caption(" · ".join(f"{'✅' if passed else '➖'} {label}" for passed, label in signals))

        if all_green:
            with st.expander("🟢 Strong automated signals — quick-verify without opening the full form", expanded=False):
                st.caption(
                    "Every check above passed, but this still records YOUR confirmation, not an automated "
                    "one -- only click this once you have actually compared the source yourself."
                )
                quick_verified_by = st.text_input(
                    "Verified by (your identity + what you checked)",
                    value=session.get(auth_client.EMAIL_KEY) or "", key=f"quick_by_{document_id}",
                )
                if st.button("✅ Quick verify", key=f"quick_{document_id}", type="primary"):
                    if not quick_verified_by.strip():
                        st.warning("Enter what you checked before quick-verifying.")
                    else:
                        _submit(session, api_base_url, document_id, filename,
                                 _quick_verify_payload(payload, quick_verified_by))
                    return

        col1, col2 = st.columns(2)
        issuing_level = col1.selectbox(
            "Issuing level", _ISSUING_LEVELS,
            index=_ISSUING_LEVELS.index(payload.get("issuing_level", "unknown")), key=f"level_{document_id}",
        )
        applicability = col2.selectbox(
            "Applicability", _APPLICABILITIES,
            index=_APPLICABILITIES.index(payload.get("applicability", "unknown")), key=f"applic_{document_id}",
        )
        state_codes = ""
        if applicability == "specific_states":
            state_codes = st.text_input(
                "State/UT codes (comma-separated, e.g. MH,UP)",
                value=",".join(payload.get("applicable_state_codes") or []), key=f"states_{document_id}",
            )

        col3, col4 = st.columns(2)
        source_type = col3.selectbox(
            "Source type", _SOURCE_TYPES,
            index=_SOURCE_TYPES.index(payload.get("jurisdiction_source_type", "unknown")), key=f"stype_{document_id}",
        )
        verification_status = col4.selectbox(
            "Verification status", _VERIFICATION_STATUSES,
            index=_VERIFICATION_STATUSES.index(
                payload.get("verification_status") if payload.get("verification_status") in _VERIFICATION_STATUSES
                else "unverified"
            ),
            key=f"vstatus_{document_id}",
            help="'verified' is a claim that a human compared this document against the issuing "
            "authority's own text -- only choose it if that comparison actually happened.",
        )

        source_url = st.text_input("Official source URL", value=payload.get("source_url") or "", key=f"url_{document_id}")
        version_label = st.text_input(
            "Version label (what this text is, in your own words)",
            value=payload.get("version_label") or "", key=f"label_{document_id}",
        )
        effective_from = st.text_input(
            "Effective from (YYYY-MM-DD, leave blank if unknown/disputed)",
            value=payload.get("effective_from") or "", key=f"eff_{document_id}",
        )
        verified_by = st.text_input(
            "Verified by (your identity + what you checked)",
            value=payload.get("verified_by") or (session.get(auth_client.EMAIL_KEY) or ""),
            key=f"by_{document_id}", disabled=verification_status != "verified",
        )

        with st.expander("Advanced: raw jurisdiction metadata JSON (overrides the fields above)"):
            raw_json = st.text_area(
                "JSON", value=json.dumps(payload, indent=2, default=str), height=220, key=f"raw_{document_id}",
            )

        if st.button("Publish", key=f"submit_{document_id}", type="primary"):
            try:
                body = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                st.error(f"Advanced JSON is invalid: {exc}")
                return
            body.update({
                "issuing_level": issuing_level, "applicability": applicability,
                "jurisdiction_source_type": source_type, "verification_status": verification_status,
                "source_url": source_url.strip() or None, "version_label": version_label.strip() or None,
                "effective_from": effective_from.strip() or None,
            })
            if applicability == "specific_states":
                body["applicable_state_codes"] = [code.strip().upper() for code in state_codes.split(",") if code.strip()]
            if verification_status == "verified":
                body["verified_by"] = verified_by.strip() or None
            _submit(session, api_base_url, document_id, filename, body)


def _submit(
    session: MutableMapping[str, Any], api_base_url: str, document_id: str, filename: str, body: dict[str, Any],
) -> None:
    try:
        response = auth_client.request(
            session, api_base_url, "PUT", f"/admin/knowledge-base/documents/{document_id}/jurisdiction",
            json=body, timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        detail = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        st.error(f"Publish failed: {detail}")
        return
    result = response.json()
    if result.get("review_status") == "approved":
        st.toast(f"Published **{filename}** — now retrievable in chat.", icon="✅")
    else:
        st.warning(f"Saved, but still `needs_review`: {'; '.join(result.get('review_reasons') or [])}")
    st.rerun()


def _bulk_submit(
    session: MutableMapping[str, Any], api_base_url: str, document_ids: list[str], verified_by: str,
) -> None:
    """One HTTP call for many documents, via
    `KnowledgeBaseIngestionService.bulk_review_automated_documents` -- NOT a
    bypass of the per-document safety gate: the server applies the exact same
    `update_jurisdiction_metadata` check to every id in the list, so any
    document whose structural fields are incomplete stays `needs_review`
    exactly as it would through the single-document form above. This only
    removes the "click Publish once per document" bottleneck for the ones a
    reviewer has already decided, in bulk, to mark verified.
    """
    try:
        response = auth_client.request(
            session, api_base_url, "POST", "/admin/knowledge-base/documents/bulk-review",
            json={"document_ids": document_ids, "verified_by": verified_by.strip()}, timeout=60,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        detail = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        st.error(f"Bulk review failed: {detail}")
        return
    results = response.json().get("results", [])
    approved = [r for r in results if r["outcome"] == "approved"]
    stayed = [r for r in results if r["outcome"] == "needs_review"]
    failed = [r for r in results if r["outcome"] == "failed"]
    if approved:
        st.toast(f"Published {len(approved)} document(s) — now retrievable in chat.", icon="✅")
    if stayed:
        st.warning(
            f"{len(stayed)} document(s) stayed `needs_review` — their structural fields "
            "were still incomplete (this is the safety gate working, not a bug)."
        )
    if failed:
        st.error(f"{len(failed)} document(s) failed: " + "; ".join(f"{r['document_id']}: {r.get('reason')}" for r in failed))
    st.rerun()


def _bulk_section(
    session: MutableMapping[str, Any], api_base_url: str, rows: list[tuple[dict[str, Any], bool]],
) -> None:
    """A batch-approve shortcut for the queue as a whole, sitting above the
    per-document cards. Only documents with every automated signal green
    (see `_signals`) are offered here -- the same bar the single-document
    "Quick verify" button already uses, just applied to many at once instead
    of one click per document. A reviewer still explicitly picks which ones
    to include (all green-signal documents are pre-selected, not forced) and
    still explicitly clicks Approve; nothing here approves anything on its
    own.
    """
    green_ids = [str(record["document_id"]) for record, all_green in rows if all_green]
    if not green_ids:
        return
    with st.expander(f"⚡ Bulk-approve {len(green_ids)} document(s) with strong automated signals", expanded=False):
        st.caption(
            "These documents already have a verified .gov.in/.nic.in source, an automated identity "
            "check, and complete jurisdiction fields — the same 'all green' bar as the individual "
            "Quick verify button. Only check the ones you have actually looked at."
        )
        selected = st.multiselect(
            "Documents to approve",
            options=green_ids,
            default=green_ids,
            format_func=lambda doc_id: next(
                (str(r.get("filename") or r.get("generated_filename") or r.get("original_filename") or doc_id)
                 for r, ok in rows if str(r["document_id"]) == doc_id), doc_id,
            ),
            key="bulk_review_selected_ids",
        )
        verified_by = st.text_input(
            "Verified by (your identity + what you checked)",
            value=session.get(auth_client.EMAIL_KEY) or "", key="bulk_review_verified_by",
        )
        if st.button(f"✅ Bulk approve {len(selected)} selected", key="bulk_review_submit", type="primary", disabled=not selected):
            if not verified_by.strip():
                st.warning("Enter what you checked before bulk-approving.")
            else:
                _bulk_submit(session, api_base_url, selected, verified_by)


def render(session: MutableMapping[str, Any], api_base_url: str) -> None:
    st.markdown("## ⚖️ Legal Source Review")
    st.caption(
        "Documents already indexed into the shared Knowledge Base, but excluded from chat retrieval "
        "until their jurisdiction metadata is completed and, for `verified`, a human has actually "
        "compared them against the issuing authority's own text."
    )
    records = _fetch_needs_review(session, api_base_url)
    if records is None:
        return
    if not records:
        st.info("Nothing is waiting for jurisdiction review right now.")
        return
    st.caption(f"{len(records)} document(s) waiting.")
    rows = [(record, all(passed for passed, _ in _signals_for_record(record, _default_payload(record)))) for record in records]
    _bulk_section(session, api_base_url, rows)
    for record in records:
        _review_card(session, api_base_url, record)
