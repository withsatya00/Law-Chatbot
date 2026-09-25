"""Chat-message rendering helpers for workflow responses.

CHAT-FIRST REFACTOR
-------------------
This module used to hold ~490 lines of Streamlit forms and dashboards: a
"Prepare for Notarization" form, an e-sign consent panel, a notary review
queue, an admin console, and a public verification page -- each reached from
its own sidebar button.

All of that is gone, not hidden. Those capabilities are now conversational:
the user asks for them in chat and `app/chatops` runs the same backend
services the forms called. What remains here is only the presentation logic
for rendering a workflow's structured result INSIDE an assistant message.

The one rule this module still owns, and the reason it stays a shared module
rather than becoming inline formatting: the status vocabulary. "Notarized"
appears in exactly one entry of `_STATUS_BADGES`, and every other state is
labelled with what it actually is -- most importantly `signed`, which says
"not notarized" out loud, because that is the confusion this product must
never create.
"""

from collections.abc import Callable, MutableMapping
from typing import Any

import streamlit as st

# Only `notarized` carries the notarized badge. `signed` is deliberately a
# warning rather than a success: a signed document is the moment a user is
# most likely to assume it has been notarized when it has not.
_STATUS_BADGES: dict[str, tuple[str, str]] = {
    "draft": ("📝", "AI-generated draft"),
    "ready_for_signature": ("📋", "Notary-ready draft"),
    "signing_in_progress": ("✍️", "Signature in progress"),
    "signed": ("🖊️", "Signed — not notarized"),
    "notary_review_requested": ("📨", "Notarization requested"),
    "notary_review_in_progress": ("🔍", "Under notary review"),
    "notarized": ("✅", "Notarized"),
    "revoked": ("⛔", "Revoked"),
}

# Workflow statuses worth showing as a visible chip on the message.
_WORKFLOW_CHIPS: dict[str, tuple[str, str]] = {
    "collecting": ("⏳", "Collecting details"),
    "awaiting_confirmation": ("❓", "Waiting for your confirmation"),
    "awaiting_upload": ("📎", "Waiting for a document"),
    "awaiting_reauth": ("🔐", "Re-authentication required"),
    "paused": ("⏸️", 'Paused — say "continue" to resume'),
    "cancelled": ("🚫", "Cancelled"),
    "forbidden": ("⛔", "Not permitted for this account"),
    "failed": ("⚠️", "Could not complete"),
}


def render_status_badge(status: str) -> None:
    icon, label = _STATUS_BADGES.get(status, ("📄", "Draft"))
    if status == "notarized":
        st.success(f"{icon} {label}")
    elif status == "revoked":
        st.error(f"{icon} {label}")
    elif status == "signed":
        st.warning(f"{icon} {label}")
    else:
        st.info(f"{icon} {label}")


def render_workflow_state(data: dict[str, Any], msg_key: str) -> None:
    """Renders the orchestration fields attached to an assistant message.

    Every field is additive and optional: a plain RAG answer carries none of
    them and renders exactly as it always did.
    """
    status = data.get("workflow_status")
    if status in _WORKFLOW_CHIPS:
        icon, label = _WORKFLOW_CHIPS[status]
        st.caption(f"{icon} {label}")

    for warning in data.get("warnings") or []:
        st.caption(f"⚠️ {warning}")

    citations = data.get("citations") or []
    if citations:
        with st.expander("Sources", expanded=False, key=f"cites_{msg_key}"):
            for citation in citations:
                title = citation.get("title", "")
                url = citation.get("url", "")
                authority = citation.get("authority", "")
                st.markdown(f"- [{title}]({url})" + (f" — _{authority}_" if authority else ""))

    facts = data.get("collected_facts") or {}
    if facts:
        with st.expander("What I have so far", expanded=False, key=f"facts_{msg_key}"):
            for key, value in facts.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")

    if data.get("upload_required"):
        st.info("📎 Attach the document using the paperclip in the message box below.")

    secure_url = data.get("secure_action_url")
    if secure_url:
        # An unavoidable third-party step (an external e-sign ceremony).
        # Rendered as a link so the user returns to this same conversation.
        st.link_button("Continue securely", secure_url)

    actions = data.get("allowed_actions") or []
    if actions:
        st.caption("You can say: " + " · ".join(f"“{action}”" for action in actions))


def render_artifact(
    artifact: dict[str, Any],
    msg_key: str,
    api_base_url: str,
    auth_state: MutableMapping[str, Any],
    request_fn: Callable[..., Any],
) -> None:
    """Renders a produced file as one or more secure inline downloads inside
    the message -- one button per format in `artifact["formats"]`, or a
    single button for `artifact["fmt"]` when `formats` is absent (every
    artifact from before this could offer more than one format at once: a
    draft export, a notarized document).

    `artifact` carries an API ROUTE plus ids -- never a filesystem path. The
    route re-checks ownership when it is called, so holding this reference is
    not itself authorisation.
    """
    formats = artifact.get("formats") or [artifact.get("fmt", "pdf")]
    if len(formats) == 1:
        _render_one_format(artifact, formats[0], msg_key, api_base_url, auth_state, request_fn)
        return
    label = artifact.get("label", "Download")
    st.caption(label)
    columns = st.columns(len(formats))
    for column, fmt in zip(columns, formats, strict=True):
        with column:
            _render_one_format(artifact, fmt, msg_key, api_base_url, auth_state, request_fn, button_label=fmt.upper())


def _render_one_format(
    artifact: dict[str, Any],
    fmt: str,
    msg_key: str,
    api_base_url: str,
    auth_state: MutableMapping[str, Any],
    request_fn: Callable[..., Any],
    button_label: str | None = None,
) -> None:
    import httpx

    label = button_label or artifact.get("label", "Download")
    cache: dict[str, bytes] = st.session_state.setdefault("draft_export_cache", {})
    cache_key = f"{artifact.get('draft_id') or artifact.get('document_id')}_{fmt}"

    if cache.get(cache_key) is not None:
        st.download_button(
            f"Save {fmt.upper()}",
            data=cache[cache_key],
            file_name=f"{cache_key}.{fmt}",
            key=f"save_art_{msg_key}_{fmt}",
        )
        return

    if not st.button(label, key=f"art_{msg_key}_{fmt}"):
        return
    try:
        if artifact.get("kind") == "draft":
            response = request_fn(
                auth_state,
                api_base_url,
                "POST",
                artifact["download_path"],
                timeout=120,
                json={"draft_id": artifact.get("draft_id"), "format": fmt},
            )
        else:
            params: dict[str, str] = {"fmt": fmt}
            if artifact.get("session_id"):
                params["session_id"] = artifact["session_id"]
            response = request_fn(
                auth_state,
                api_base_url,
                "GET",
                artifact["download_path"],
                timeout=120,
                params=params,
            )
        if response.status_code == 200:
            cache[cache_key] = response.content
            st.rerun()
        else:
            st.error("That file could not be generated. Please try again.")
    except httpx.RequestError:
        st.error("Could not reach the server to generate that file.")
