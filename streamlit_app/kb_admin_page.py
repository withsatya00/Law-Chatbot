"""Admin-only Knowledge Base review page.

`app.py`'s own "CHAT-FIRST REFACTOR" comment documents a deliberate decision
to reach every admin capability by talking rather than through dedicated
pages -- the same approve/reject actions already work by chat (see
`app/chatops/workflows/admin.py`). This page is a requested, explicit
exception to that for one thing chat can't do well: showing the actual PDF
content next to the approve/reject decision. It calls the exact same
`/admin/knowledge-base/staging/*` routes the chat workflow calls, so nothing
here bypasses role checks, `require_admin`, or the audit log.
"""

from collections.abc import MutableMapping
from typing import Any

import auth_client
import httpx
import streamlit as st

STATUS_LABELS = {
    "pending": "⏳ Pending",
    "processing": "⚙️ Processing",
    "indexed": "✅ Indexed",
    "duplicate": "♻️ Duplicate",
    "failed": "❌ Failed",
    "needs_review": "🔎 Needs review",
}
_FILTERS = ["all", "needs_review", "pending", "processing", "indexed", "duplicate", "failed"]


def _fetch_records(session: MutableMapping[str, Any], api_base_url: str, status: str) -> list[dict[str, Any]] | None:
    try:
        response = auth_client.request(
            session, api_base_url, "GET", "/admin/knowledge-base/staging",
            params={"status": status}, timeout=20,
        )
        response.raise_for_status()
        return response.json().get("records", [])
    except httpx.HTTPError:
        st.error("Could not reach the server to load the Knowledge Base staging list.")
        return None


def _fetch_file_bytes(session: MutableMapping[str, Any], api_base_url: str, staging_id: str) -> bytes | None:
    # Cached for the life of this browser session -- an admin scrolling
    # through the queue should not re-download the same PDF on every rerun.
    cache: dict[str, bytes] = st.session_state.setdefault("kb_admin_pdf_cache", {})
    if staging_id in cache:
        return cache[staging_id]
    try:
        response = auth_client.request(
            session, api_base_url, "GET", f"/admin/knowledge-base/staging/{staging_id}/file", timeout=60,
        )
        if response.status_code != 200:
            return None
        cache[staging_id] = response.content
        return response.content
    except httpx.HTTPError:
        return None


def _preview(session: MutableMapping[str, Any], api_base_url: str, record: dict[str, Any], scope: str) -> None:
    """`scope` namespaces this call's widget keys ("review" vs "all") -- the
    same record can legitimately render in both tabs in the same script run
    (Streamlit renders every tab's content, not just the active one), so a
    key built from the staging id alone would collide across tabs."""
    staging_id = str(record["_id"])
    filename = str(record.get("original_filename") or "file")
    if not filename.lower().endswith(".pdf"):
        st.caption(f"📎 {filename} — preview is only available for PDFs.")
        return
    widget_key = f"kb_admin_{scope}_{staging_id}"
    # The loaded flag itself is intentionally shared across scopes (loading it
    # once in either tab keeps it expanded in both) -- only the button/
    # download widget keys below need to be scope-unique.
    loaded_key = f"kb_admin_loaded_{staging_id}"
    if not st.session_state.get(loaded_key):
        if st.button("👁️ Load preview", key=f"kb_admin_load_{widget_key}"):
            st.session_state[loaded_key] = True
        else:
            return
    data = _fetch_file_bytes(session, api_base_url, staging_id)
    if data is None:
        st.caption("Preview unavailable — the file may be missing on disk.")
        return
    try:
        st.pdf(data, height=420)
    except Exception:  # noqa: BLE001 - a preview-only component must never break the page
        st.caption("Preview unavailable for this PDF.")
        st.download_button("Download PDF", data=data, file_name=filename, key=f"kb_admin_dl_{widget_key}")


def _forget(staging_id: str) -> None:
    st.session_state.pop(f"kb_admin_loaded_{staging_id}", None)
    st.session_state.get("kb_admin_pdf_cache", {}).pop(staging_id, None)


def _approve(session: MutableMapping[str, Any], api_base_url: str, staging_id: str, filename: str) -> None:
    try:
        response = auth_client.request(
            session, api_base_url, "POST", f"/admin/knowledge-base/staging/{staging_id}/approve", timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Approval failed: {exc}")
        return
    _forget(staging_id)
    st.toast(f"Approved **{filename}** — queued for indexing.", icon="✅")
    st.rerun()


def _reject(session: MutableMapping[str, Any], api_base_url: str, staging_id: str, filename: str, reason: str) -> None:
    try:
        response = auth_client.request(
            session, api_base_url, "POST", f"/admin/knowledge-base/staging/{staging_id}/archive",
            params={"reason": reason}, timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Rejection failed: {exc}")
        return
    _forget(staging_id)
    st.toast(f"Rejected **{filename}**.", icon="🚫")
    st.rerun()


def _close_missing(
    session: MutableMapping[str, Any], api_base_url: str, staging_id: str, filename: str
) -> None:
    reason = "Closed stale Knowledge Base review record because its source file is missing from disk."
    try:
        response = auth_client.request(
            session,
            api_base_url,
            "POST",
            f"/admin/knowledge-base/staging/{staging_id}/close-missing",
            params={"reason": reason},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Could not close the stale record: {exc}")
        return
    _forget(staging_id)
    st.toast(f"Closed stale record for **{filename}**.")
    st.rerun()


def _needs_review_card(session: MutableMapping[str, Any], api_base_url: str, record: dict[str, Any]) -> None:
    staging_id = str(record["_id"])
    filename = str(record.get("original_filename") or "file")
    file_exists = bool(record.get("file_exists", True))
    with st.container(border=True):
        st.markdown(f"**📄 {filename}**")
        st.caption(
            f"Staging id `{staging_id}` · reason: {record.get('reason') or '—'} · "
            + ("file present" if file_exists else "⚠️ **file missing from disk**")
        )
        if not file_exists:
            st.info(
                "This file cannot be previewed or approved because its bytes are no longer on disk. "
                "Remove the stale queue record, then upload the source again if it is still needed."
            )
            if st.button(
                "Remove stale record",
                key=f"kb_admin_remove_stale_{staging_id}",
                type="primary",
            ):
                _close_missing(session, api_base_url, staging_id, filename)
            return
        _preview(session, api_base_url, record, scope="review")
        approve_col, reject_col = st.columns([1, 2])
        if approve_col.button("✅ Approve", key=f"kb_admin_approve_{staging_id}", type="primary", disabled=not file_exists):
            _approve(session, api_base_url, staging_id, filename)
        reason = reject_col.text_input(
            "Rejection reason", key=f"kb_admin_reason_{staging_id}",
            label_visibility="collapsed", placeholder="Reason (required to reject)",
        )
        if reject_col.button("🚫 Reject", key=f"kb_admin_reject_{staging_id}"):
            if not reason.strip():
                st.warning("A reason is required to reject.")
            else:
                _reject(session, api_base_url, staging_id, filename, reason.strip())


def _all_files_row(session: MutableMapping[str, Any], api_base_url: str, record: dict[str, Any]) -> None:
    filename = str(record.get("original_filename") or "file")
    status = str(record.get("status") or "")
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].markdown(f"**📄 {filename}**")
        cols[1].markdown(STATUS_LABELS.get(status, status))
        if record.get("reason"):
            st.caption(f"Reason: {record['reason']}")
        if not record.get("file_exists", True):
            st.caption("⚠️ File missing from disk")
        _preview(session, api_base_url, record, scope="all")


def render(session: MutableMapping[str, Any], api_base_url: str) -> None:
    st.markdown("## 🗂️ Knowledge Base Review")
    st.caption(
        "Every PDF staged in the Knowledge Base pipeline. Approve or reject the ones waiting on "
        "review here — the same actions the chat's \"needs review\" flow performs."
    )

    tab_review, tab_all = st.tabs(["🔎 Needs review", "📚 All files"])

    with tab_review:
        records = _fetch_records(session, api_base_url, "needs_review")
        if records is None:
            pass
        elif not records:
            st.info("Nothing is waiting for review right now.")
        else:
            st.caption(f"{len(records)} file(s) waiting for review.")
            for record in records:
                _needs_review_card(session, api_base_url, record)

    with tab_all:
        status_choice = st.selectbox(
            "Filter by status", _FILTERS, key="kb_admin_status_filter",
            format_func=lambda s: "All statuses" if s == "all" else STATUS_LABELS.get(s, s),
        )
        records = _fetch_records(session, api_base_url, status_choice)
        if records is None:
            pass
        elif not records:
            st.info("No files with this status.")
        else:
            st.caption(f"{len(records)} file(s).")
            for record in records:
                _all_files_row(session, api_base_url, record)
