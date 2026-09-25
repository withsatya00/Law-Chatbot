import base64
import hashlib
import json
import os
import re
import time
from collections.abc import MutableMapping
from typing import Any, cast
from uuid import uuid4

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# `st.session_state` behaves as a mutable mapping but is not declared as one
# in Streamlit's stubs, so it is narrowed ONCE here rather than at each of the
# dozen call sites that hand it to `auth_client`. Same object, one cast.
SESSION: MutableMapping[str, Any] = cast(MutableMapping[str, Any], st.session_state)
CLIENT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("CLIENT_REQUEST_TIMEOUT_SECONDS", "180"))

# E-Notarization UI (prepare/e-sign/notary queue/admin/verification).
# Kept in its own module rather than inlined here -- see
# `streamlit_app/notarization_ui.py`.
import auth_client
import coverage_admin_page
import document_extraction_ui
import kb_admin_page
import legal_source_review_page
import notarization_ui
from answer_apparatus import source_lines, suppresses_apparatus
from kb_upload_status import TERMINAL_UPLOAD_STATUSES, status_note
from upload_routing import UploadTarget, upload_error_note, upload_target

LANGUAGES = [
    "Auto", "English", "Hindi", "Hinglish", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Gujarati", "Marathi", "Punjabi", "Bengali", "Odia", "Urdu", "Assamese",
    "Bodo", "Dogri", "Kashmiri", "Konkani", "Maithili", "Manipuri", "Nepali",
    "Sanskrit", "Santali", "Sindhi",
]
UPLOAD_FILE_TYPES = ["pdf", "docx", "txt", "md", "html", "json", "csv", "rtf", "odt", "png", "jpg", "jpeg"]
EXAMPLE_PROMPTS = [
    "What are my rights if my landlord won't return my security deposit?",
    "Draft a rental agreement for a 2BHK flat in Bengaluru",
    "Explain Section 138 of the Negotiable Instruments Act",
    "What should I do after a cheque bounce notice?",
]


def _repair_mojibake(value: Any) -> Any:
    """Repair UTF-8 bytes accidentally decoded as a Windows/Latin encoding.

    The conversion is deliberately conservative: ordinary text is returned
    untouched, while strings carrying the characteristic mojibake markers
    used by broken Hindi/emoji/currency text are round-tripped once. The
    recursive handling keeps API answers, draft metadata and source labels in
    the same encoding state.
    """
    if isinstance(value, dict):
        return {key: _repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_repair_mojibake(item) for item in value]
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "Â", "â", "ð", "à¤", "à¥")):
        return value
    for encoding in ("cp1252", "latin1"):
        try:
            repaired = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired.count("�") == 0:
            return repaired
    return value

st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="centered", initial_sidebar_state="expanded")

st.html("""
<style>
[data-testid="stMainBlockContainer"] { max-width: 820px; padding-top: 1.5rem; padding-bottom: 6rem; }
#MainMenu, [data-testid="stAppDeployButton"] { visibility: hidden; }
footer { visibility: hidden; }

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.4rem;
    max-width: 90%;
    gap: 0.6rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--primary-color);
    margin-left: auto;
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {
    color: var(--background-color) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--secondary-background-color);
    margin-right: auto;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.3rem; }

/* Part 39 section 16 "UI Typography": chat replies were rendering raw
   browser default heading sizes (H1 ~2rem+), which read as jarring,
   inconsistent shouting next to normal chat text. Scoped to message
   content only -- the sidebar/page chrome uses its own H2 (see
   `_new_chat` welcome heading below) and shouldn't shrink with this. */
[data-testid="stChatMessageContent"] {
    line-height: 1.6;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    font-size: 1rem;
    line-height: 1.6;
}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2 {
    font-size: 1.6rem;
    margin: 16px 0 8px 0;
}
[data-testid="stChatMessageContent"] h3 {
    font-size: 1.2rem;
    margin: 16px 0 8px 0;
}
[data-testid="stChatMessageContent"] h1:first-child,
[data-testid="stChatMessageContent"] h2:first-child,
[data-testid="stChatMessageContent"] h3:first-child {
    margin-top: 0;
}
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol {
    margin: 8px 0;
}
[data-testid="stChatMessageContent"] table {
    display: block;
    overflow-x: auto;
    max-width: 100%;
}

/* Post-Phase-3 hardening (Phase 2, milestone E) -- message whitespace.

   An assistant turn is not one thing. It is an answer, sometimes workflow
   progress, sometimes a draft preview, then sources, warnings and download
   controls. All of it was emitted as a flat sequence of Streamlit blocks
   separated only by the default vertical gap, which on a real reply (answer +
   preview + chips + downloads + feedback row) leaves several screens of
   near-empty space and no way to tell where the assistant's answer ends and
   the machinery around it begins.

   Two changes, both purely presentational: tighten the default gaps inside a
   message, and give the trailing apparatus a labelled band of its own (see
   `_render_answer_apparatus`). No feature buttons and no separate pages are
   reintroduced -- everything stays inside the one conversation. */
[data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:empty {
    display: none;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessage"] [data-testid="stExpander"] summary { padding: 0.25rem 0.6rem; }
[data-testid="stChatMessage"] [data-testid="stExpander"] p { margin-bottom: 0.25rem; }

/* The apparatus band: a quiet rule and a smaller type scale, so it reads as
   secondary to the answer above it rather than as more answer. */
.answer-apparatus-rule {
    border: 0;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    margin: 0.55rem 0 0.35rem 0;
}
.apparatus-label {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.62;
    margin: 0.15rem 0 0.1rem 0;
}
</style>
""")


def _new_chat() -> dict[str, Any]:
    # Part 45 "Per-User Document Isolation": generated up front (like "id"
    # already was), not left `None` until the first `/chat` round-trip --
    # an upload made before any chat message needs a session_id to attach
    # to the document so it isn't globally visible to other sessions. The
    # backend already accepts a client-supplied session_id as-is, so this
    # needs no backend change.
    return {"id": str(uuid4()), "title": "", "session_id": str(uuid4()), "messages": []}


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    chat = _new_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.active_chat_id = chat["id"]
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False
if "admin_view" not in st.session_state:
    st.session_state.admin_view = "chat"


def _active_chat() -> dict[str, Any]:
    return st.session_state.chats[st.session_state.active_chat_id]


def _typing_stream(text: str):
    words = text.split(" ")
    delay = min(0.03, max(0.006, 2.2 / max(len(words), 1)))
    for index, word in enumerate(words):
        yield word + (" " if index < len(words) - 1 else "")
        time.sleep(delay)


# "all user/admin capabilities continue to be invoked through the chat" (see
# the sidebar comment above) rules out a separate jurisdiction-metadata form
# widget -- an admin instead appends one JSON object to the same message that
# triggers the KB upload, e.g. `add to kb {"issuing_level": "state",
# "applicability": "specific_states", "applicable_state_codes": ["MH"]}`.
# Greedy first-"{"-to-last-"}" is enough here: the admin controls the whole
# message and puts exactly one object at the end; nothing needs to survive
# other braces appearing elsewhere in free text.
_JURISDICTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _split_jurisdiction_metadata(text: str, target: UploadTarget) -> tuple[str, str | None]:
    """Returns `(display_text_with_json_removed, jurisdiction_json_or_None)`.
    Only looked for on an `admin_kb` upload -- a private attachment's message
    text is never parsed as anything but the user's own words."""
    if target != "admin_kb":
        return text, None
    match = _JURISDICTION_JSON_RE.search(text)
    if not match:
        return text, None
    candidate = match.group(0)
    try:
        json.loads(candidate)
    except ValueError:
        return text, None
    remaining = (text[: match.start()] + text[match.end() :]).strip()
    return remaining, candidate


def _upload_attachments(
    files: list,
    session_id: str | None,
    target: UploadTarget = "private",
    jurisdiction_metadata_json: str | None = None,
) -> list[dict[str, Any]]:
    attachments = []
    for uploaded in files:
        data = uploaded.getvalue()
        if target == "admin_required":
            attachments.append(
                {
                    "name": uploaded.name,
                    "type": uploaded.type or "",
                    "data": data,
                    "note": "Not uploaded. Sign in with an administrator account to add a shared KB source.",
                    "uploaded": False,
                }
            )
            continue
        try:
            kb_status: dict[str, Any] | None = None
            path = "/admin/knowledge-base/upload" if target == "admin_kb" else "/upload"
            form_data: dict[str, Any] | None = None
            if session_id and target == "private":
                form_data = {"session_id": session_id}
            elif target == "admin_kb" and jurisdiction_metadata_json:
                # Left out entirely (rather than sent empty) when the admin
                # supplied nothing -- the API still defaults an omitted field
                # to needs_review, so no metadata is not a way to skip review.
                form_data = {"jurisdiction_metadata": jurisdiction_metadata_json}
            response = auth_client.request(SESSION, API_BASE_URL, "POST", path, timeout=630,
                files={"file": (uploaded.name, data, uploaded.type or "application/octet-stream")},
                data=form_data,
            )
            if response.status_code == 200:
                payload = response.json()
                if target == "admin_kb":
                    kb_status = payload
                    note = status_note(payload)
                else:
                    chunks = payload.get("chunks_indexed", 0)
                    note = f"Uploaded for this conversation ({chunks} extracted section{'s' if chunks != 1 else ''})."
                    # Surfaces what was previously only a server-side log line
                    # (a scanned PDF whose OCR was unavailable/disabled or
                    # added nothing over its near-empty embedded text) -- see
                    # `UploadResponse.warnings` / `DocumentLoader._load_pdf`.
                    for warning in payload.get("warnings") or []:
                        note += f" ⚠️ {warning}"
            else:
                note = upload_error_note(response)
        except httpx.RequestError:
            note = "Could not reach the server to index this file."
        attachment = {"name": uploaded.name, "type": uploaded.type or "", "data": data, "note": note,
                      "uploaded": note.startswith("Uploaded for this conversation") or kb_status is not None}
        if kb_status is not None:
            attachment["kb_staging_id"] = kb_status.get("staging_id")
            attachment["kb_status"] = kb_status.get("status", "pending")
        attachments.append(attachment)
    return attachments


def _refresh_kb_attachment_status(attachment: dict[str, Any]) -> None:
    staging_id = attachment.get("kb_staging_id")
    current = str(attachment.get("kb_status") or "")
    if current in TERMINAL_UPLOAD_STATUSES:
        return
    if staging_id:
        path = f"/admin/knowledge-base/staging/{staging_id}"
    elif str(attachment.get("note") or "").startswith("Submitted to shared Knowledge Base"):
        # Receipts created before upload tracking IDs were added can still be
        # resolved safely by the exact bytes already stored in this chat.
        content_hash = hashlib.sha256(attachment.get("data") or b"").hexdigest()
        path = f"/admin/knowledge-base/content/{content_hash}/status"
    else:
        return
    try:
        response = auth_client.request(SESSION,
            API_BASE_URL,
            "GET",
            path,
            timeout=5,
        )
        if response.status_code != 200:
            return
        payload = response.json()
        attachment["kb_staging_id"] = payload.get("staging_id")
        attachment["kb_status"] = payload.get("status", current)
        attachment["note"] = status_note(payload)
    except (httpx.HTTPError, ValueError):
        # Status refresh is supplementary; keep the last truthful receipt if
        # the API is temporarily unavailable or returns malformed data.
        return


def _render_attachment(attachment: dict[str, Any]) -> None:
    _refresh_kb_attachment_status(attachment)
    name, mime, data = attachment["name"], attachment.get("type", ""), attachment["data"]
    if mime.startswith("image/"):
        st.image(data, width=220, caption=name)
    elif mime == "application/pdf" or name.lower().endswith(".pdf"):
        with st.expander(f"📄 {name}", expanded=False):
            try:
                st.pdf(data, height=380)
            except Exception:  # noqa: BLE001 - a preview-only component must never break the page (see comment below)
                # The streamlit-pdf component has a known intermittent
                # registration failure (StreamlitAPIException from a
                # manifest-scan ordering issue, not anything wrong with this
                # specific file) -- never let a preview-only feature break
                # the chat turn; just offer the file as a download instead.
                st.caption("Preview unavailable for this PDF.")
                st.download_button("Download PDF", data=data, file_name=name, key=f"pdfdl_{name}_{len(data)}")
    else:
        st.caption(f"📎 {name}")
    if attachment.get("note"):
        st.caption(attachment["note"])


def _send_chip(label: str) -> None:
    st.session_state.pending_input = label
    st.rerun()


def _render_chips(prefix: str, items: list[str]) -> None:
    if not items:
        return
    cols = st.columns(len(items)) if len(items) <= 4 else st.columns(4)
    for index, item in enumerate(items):
        with cols[index % len(cols)]:
            if st.button(item, key=f"{prefix}_{index}", width="stretch"):
                _send_chip(item)


# Problem-first draft discovery -- four entry actions, per the spec:
# "Describe my legal problem" (the default -- a generic drafting request
# just seeds that exact chat message, which `DraftConversationEngine`'s new
# discovery flow picks up and asks 1-2 targeted questions instead of ever
# listing all ~55 templates), "Search a draft" and "Browse categories" (both
# bounded, ranked API calls -- never render the whole catalogue), and
# "Continue saved draft" (routes into the existing "saved_drafts" chatops
# workflow, unchanged by this feature). Shown only on the empty-chat screen,
# same as `EXAMPLE_PROMPTS` below -- this is not a new page, just four more
# buttons that seed the one conversation, consistent with the "CHAT-FIRST
# REFACTOR" this file already committed to (see the comment block near the
# bottom of this file).
_DRAFT_SEARCH_RESULT_LIMIT = 5
_DRAFT_CATEGORY_TEMPLATE_LIMIT = 8


def _render_draft_entry_actions(session_id: str) -> None:
    st.caption("Draft a document")
    describe_col, search_col, browse_col, saved_col = st.columns(4)
    with describe_col:
        if st.button("📝 Describe my legal problem", key="draft_entry_describe", width="stretch", type="primary"):
            _send_chip("Mujhe draft generate karni hai")
    with search_col:
        with st.popover("🔍 Search a draft", width="stretch"):
            query = st.text_input("Document name", key="draft_search_query", placeholder="e.g. cheque bounce notice")
            if query.strip():
                try:
                    response = auth_client.request(
                        SESSION, API_BASE_URL, "GET", "/draft-templates/search",
                        timeout=15, params={"q": query.strip()},
                    )
                    response.raise_for_status()
                    results = response.json().get("results", [])[:_DRAFT_SEARCH_RESULT_LIMIT]
                except httpx.HTTPError:
                    results = []
                    st.caption("Search is unavailable right now.")
                for result in results:
                    label = f"{result['name']} / {result['hindi_name']}" if result.get("hindi_name") else result["name"]
                    if st.button(label, key=f"draft_search_result_{result['draft_id']}", width="stretch"):
                        _send_chip(f"{result['name']} banao")
                if query.strip() and not results:
                    st.caption("No matching draft found -- try describing your problem instead.")
    with browse_col:
        with st.popover("📂 Browse categories", width="stretch"):
            try:
                response = auth_client.request(SESSION, API_BASE_URL, "GET", "/draft-categories", timeout=15)
                response.raise_for_status()
                categories = response.json()
            except httpx.HTTPError:
                categories = []
                st.caption("Categories are unavailable right now.")
            chosen_category = st.session_state.get("draft_browse_category")
            if not chosen_category:
                for category in categories:
                    label = f"{category['name']} ({category['template_count']})"
                    if st.button(label, key=f"draft_category_{category['category_id']}", width="stretch"):
                        st.session_state["draft_browse_category"] = category["category_id"]
                        st.rerun()
            else:
                if st.button("← Back to categories", key="draft_category_back"):
                    st.session_state.pop("draft_browse_category", None)
                    st.rerun()
                try:
                    templates_response = auth_client.request(
                        SESSION, API_BASE_URL, "GET", "/draft-templates",
                        timeout=15, params={"domain": chosen_category},
                    )
                    templates_response.raise_for_status()
                    templates = templates_response.json()[:_DRAFT_CATEGORY_TEMPLATE_LIMIT]
                except httpx.HTTPError:
                    templates = []
                if not templates:
                    st.caption("No templates are organized under this category yet -- try search instead.")
                for template in templates:
                    if st.button(template["name"], key=f"draft_cat_tpl_{template['draft_id']}", width="stretch"):
                        st.session_state.pop("draft_browse_category", None)
                        _send_chip(f"{template['name']} banao")
    with saved_col:
        if st.button("📁 Continue saved draft", key="draft_entry_saved", width="stretch"):
            _send_chip("Meri saved drafts dikhao")


def _render_draft_downloads(draft_info: dict[str, Any], msg_key: str, session_id: str) -> None:
    draft_id = draft_info.get("draft_id")
    if not draft_id:
        return
    # The backend permits export as soon as the preview exists. Keep the UI
    # in the same lifecycle model so a valid preview can be downloaded.
    if draft_info.get("stage") not in ("preview", "approved", "locked", "exported"):
        return
    st.divider()
    st.caption(f"📝 Draft: {draft_info.get('template_name', '')}")
    formats = draft_info.get("available_export_formats") or []
    if not formats:
        return
    # The exported file bytes used to live only in the local variable
    # produced by THIS SPECIFIC script run's button click -- any other rerun
    # in between (a chip click, a sidebar toggle, even another export button
    # elsewhere on the page) reset the whole script and threw the fetched
    # bytes away, silently reverting the "Save {fmt}" button back to
    # "Download {fmt}" with no error shown. That's indistinguishable, from
    # the user's side, from the download simply not working. Caching the
    # bytes in `session_state` (keyed by draft_id+format, so a re-export
    # after "unlock" -> edit -> re-lock correctly fetches fresh bytes rather
    # than serving the stale pre-edit file) makes the "Save" button survive
    # any number of unrelated reruns once fetched once.
    export_cache: dict[str, bytes] = st.session_state.setdefault("draft_export_cache", {})
    download_cols = st.columns(len(formats))
    for col, fmt in zip(download_cols, formats):
        with col:
            cache_key = f"{draft_id}_{fmt}"
            cached_bytes = export_cache.get(cache_key)
            if cached_bytes is not None:
                st.download_button(
                    f"Save {fmt.upper()}",
                    data=cached_bytes,
                    file_name=f"{draft_id}.{fmt}",
                    key=f"save_{msg_key}_{fmt}",
                )
            elif st.button(f"Download {fmt.upper()}", key=f"dl_{msg_key}_{fmt}"):
                try:
                    export_payload = {"draft_id": draft_id, "format": fmt, "session_id": session_id}
                    file_response = auth_client.request(
                        SESSION, API_BASE_URL, "POST", "/draft/export", timeout=120, json=export_payload,
                    )
                    if file_response.status_code == 404:
                        # A reconnect immediately after startup can make one
                        # lookup miss. Retry once; a repeated 404 is a stale
                        # draft reference, not a reason to loop forever.
                        time.sleep(0.35)
                        file_response = auth_client.request(
                            SESSION, API_BASE_URL, "POST", "/draft/export", timeout=120, json=export_payload,
                        )
                    if file_response.status_code == 200:
                        export_cache[cache_key] = file_response.content
                        st.rerun()
                    else:
                        try:
                            error_message = file_response.json().get("error", {}).get("message", "")
                        except ValueError:
                            error_message = ""
                        if file_response.status_code == 404:
                            draft_info["available_export_formats"] = []
                            st.error(
                                "This draft is no longer available on the server. Regenerate or reopen the "
                                "current saved draft before downloading."
                            )
                        else:
                            st.error(error_message or f"Couldn't generate the {fmt.upper()} right now.")
                except httpx.RequestError:
                    st.error(f"Couldn't reach the server to generate the {fmt.upper()}.")

    # Preparing a draft for notarization is conversational now -- the user
    # says "is draft ko notary ke liye prepare karo" and `app/chatops`
    # handles it. There is no form here any more.


def _render_dev_panel(data: dict[str, Any], msg_key: str) -> None:
    with st.expander("🛠️ Developer details", expanded=False, key=f"dev_{msg_key}"):
        cols = st.columns(3)
        cols[0].metric(f"Confidence: {data.get('confidence_label', '—')}", data.get("confidence", 0))
        cols[1].metric("Latency (ms)", round(data.get("latency_ms", 0), 1))
        cols[2].metric("Language", data.get("detected_language", "—"))
        st.write("Detected intent:", data.get("detected_intent"))
        st.write("Conversation intent:", data.get("conversation_intent"))
        st.write("LLM:", f"{data.get('llm_provider', '—')} / {data.get('llm_model', '—')}")
        st.write("Prompt version:", data.get("prompt_version", "—"))
        if data.get("extracted_entities"):
            st.markdown("**Entities**")
            st.json(data["extracted_entities"])
        if data.get("sources"):
            # The readable citation list moved out to `_render_answer_apparatus`,
            # where every user sees it. What stays here is the raw record, which
            # is what a developer opening this panel actually wants.
            with st.expander("Source details"):
                st.json(data["sources"])
        if data.get("retrieved_chunks"):
            st.markdown("**Retrieved chunks**")
            st.json(data["retrieved_chunks"])


def _render_voice_playback(data: dict[str, Any], msg_key: str, answer_text: str) -> None:
    """Plays back audio for an assistant reply -- either audio `/voice/chat`
    already generated (stored on `data["audio_base64"]`, a voice-in/
    voice-out turn), or, for an ordinary typed-question reply, a lazy
    "Read aloud" button that fetches it on demand from `/voice/speak` and
    caches the result back onto `data` so a second click doesn't re-call
    the API for audio already fetched once.
    """
    audio_b64 = data.get("audio_base64")
    if audio_b64:
        st.audio(base64.b64decode(audio_b64), format="audio/wav")
        return
    if st.button("🔊 Read aloud", key=f"speak_{msg_key}"):
        try:
            response = auth_client.request(SESSION, API_BASE_URL, "POST", "/voice/speak",
                timeout=30, json={"text": answer_text},
            )
            response.raise_for_status()
            data["audio_base64"] = response.json().get("audio_base64")
        except httpx.HTTPError:
            st.caption("Couldn't generate audio right now — please try again.")
            return
        st.rerun()


def _render_answer_apparatus(data: dict[str, Any], msg_key: str) -> None:
    """Warnings, sources and currency notes, in one visually separated band.

    Order is deliberate: a warning changes what the reader should do with the
    answer, so it comes first; the sources come next because they are what a
    reader checks the answer against; the currency note last because it
    qualifies the sources rather than the answer.
    """
    warnings = [str(item) for item in (data.get("warnings") or []) if item]
    # A no-verified-context refusal carries no support by construction, so
    # none of this band applies to it. The API already empties these fields
    # (`app/services/safe_decline.py`); this also covers a legacy payload
    # replayed from history that still has them.
    suppressed = suppresses_apparatus(data)
    currency = "" if suppressed else str(data.get("currency_notice") or "").strip()
    sources = source_lines(data)
    if not (warnings or currency or sources):
        return
    st.html('<hr class="answer-apparatus-rule">')
    for warning in warnings:
        st.warning(warning, icon="⚠️")
    if sources:
        st.html('<div class="apparatus-label">Sources</div>')
        st.markdown("\n".join(sources))
    if currency:
        with st.expander("How current are these sources?", expanded=False, key=f"cur_{msg_key}"):
            st.markdown(currency)


def _render_assistant_extras(data: dict[str, Any] | None, msg_key: str, session_id: str, answer_text: str = "") -> None:
    if not data:
        return
    _render_voice_playback(data, msg_key, answer_text)
    # Conversational-workflow state (status chip, warnings, sources, collected
    # facts, allowed actions, inline artifact). All optional -- a plain RAG
    # answer carries none of it and renders exactly as before.
    #
    # Wrapped because these are DECORATIONS around an answer the user has
    # already been given. A helper that raises here used to abort the whole
    # turn and take the answer down with it -- which is exactly what happened
    # when a stale `notarization_ui` module (Streamlit caches imports in
    # `sys.modules` across reruns, so an edited module needs a server
    # restart) no longer had `render_workflow_state`. A cosmetic renderer
    # must never be able to destroy the conversation.
    try:
        notarization_ui.render_workflow_state(data, msg_key)
        artifact = data.get("artifact")
        if isinstance(artifact, dict):
            notarization_ui.render_artifact(
                artifact,
                msg_key,
                API_BASE_URL,
                SESSION,
                auth_client.request,
            )
    except AttributeError:
        # Almost always a stale module held from before an edit. Say what to
        # do rather than showing a traceback over the user's answer.
        st.caption("⚠️ Restart the app to load the latest interface changes.")
    except Exception:  # noqa: BLE001 - never let extras kill the answer
        st.caption("⚠️ Some extra details could not be displayed.")
    _render_answer_apparatus(data, msg_key)
    _render_chips(f"sugg_{msg_key}", data.get("suggested_actions") or [])
    _render_chips(f"rel_{msg_key}", data.get("related_questions") or [])
    draft_info = data.get("draft")
    if draft_info:
        if data.get("voice_final_confirmation_required"):
            st.warning("Review the voice-collected facts before export.")
            confirmation = st.text_input("Type I CONFIRM THE REVIEWED FACTS", key=f"voice_confirm_{msg_key}")
            if st.button("Confirm reviewed voice facts", key=f"voice_confirm_btn_{msg_key}"):
                try:
                    response = auth_client.request(SESSION,
                        API_BASE_URL,
                        "POST",
                        f"/voice/drafts/{draft_info.get('draft_id')}/confirm",
                        json={"session_id": session_id, "confirmation_text": confirmation},
                        timeout=20,
                    )
                    response.raise_for_status()
                    data["voice_final_confirmation_required"] = False
                    st.rerun()
                except httpx.HTTPError:
                    st.error("Confirmation was not accepted. Review the draft and use the exact confirmation text.")
        st.html('<div class="apparatus-label">Download this draft</div>')
        _render_draft_downloads(draft_info, msg_key, session_id)
    message_id = data.get("message_id")
    if message_id:
        feedback_labels = [
            ("Helpful", "helpful", 5), ("Wrong language", "wrong_language", 1),
            ("Wrong law", "wrong_law", 1), ("Missing source", "missing_source", 1),
            ("Unsafe draft", "unsafe_draft", 1),
        ]
        cols = st.columns(len(feedback_labels))
        for col, (label, category, rating) in zip(cols, feedback_labels):
            if col.button(label, key=f"feedback_{msg_key}_{category}"):
                try:
                    auth_client.request(SESSION,
                        API_BASE_URL,
                        "POST",
                        "/feedback",
                        json={"session_id": session_id, "message_id": message_id, "rating": rating, "category": category},
                        timeout=10,
                    ).raise_for_status()
                    st.toast("Feedback recorded")
                except httpx.HTTPError:
                    st.toast("Feedback could not be saved. Please retry.")
    if st.session_state.dev_mode:
        _render_dev_panel(data, msg_key)


def _render_history(chat: dict[str, Any]) -> None:
    search_term = st.session_state.get("chat_search", "").strip().casefold()
    # A draft payload belongs to the assistant turn that created it. Older
    # turns remain useful as history, but their export controls are snapshots,
    # not the session's active draft. Only the newest drafting turn is allowed
    # to expose actionable controls; this prevents an abandoned Consumer
    # Complaint card appearing active while the user is collecting fields for
    # a later affidavit/notice.
    latest_draft_index = next(
        (
            index
            for index in range(len(chat["messages"]) - 1, -1, -1)
            if ((chat["messages"][index].get("meta") or {}).get("draft"))
        ),
        None,
    )
    for index, message in enumerate(chat["messages"]):
        content = _repair_mojibake(message.get("content", ""))
        if search_term and search_term not in content.casefold():
            continue
        with st.chat_message(message["role"]):
            st.markdown(content)
            for attachment in message.get("attachments") or []:
                _render_attachment(attachment)
            if message["role"] == "assistant":
                meta = _repair_mojibake(message.get("meta"))
                if meta and index != latest_draft_index and meta.get("draft"):
                    meta = {**meta, "draft": {**meta["draft"], "available_export_formats": []}}
                _render_assistant_extras(meta, f"h{index}", chat["session_id"], message["content"])


def _handle_turn(chat: dict[str, Any], text: str, files: list, language_choice: str) -> None:
    display_text = text.strip()
    target = upload_target(display_text, auth_client.current_role(SESSION))
    # An embedded jurisdiction-metadata JSON object is admin bookkeeping, not
    # part of the question -- stripped out of what gets shown in the chat and
    # sent to the answer pipeline, same as the file itself isn't re-asked as
    # a question. See `_split_jurisdiction_metadata`.
    display_text, jurisdiction_metadata_json = _split_jurisdiction_metadata(display_text, target)
    attachments = (
        _upload_attachments(files, chat["session_id"], target, jurisdiction_metadata_json) if files else []
    )
    if not display_text and attachments:
        names = ", ".join(attachment["name"] for attachment in attachments)
        display_text = f"Summarize the uploaded document: {names}. Explain its key points."

    chat["messages"].append({"role": "user", "content": display_text, "attachments": attachments})
    if not chat["title"]:
        chat["title"] = display_text[:48] + ("…" if len(display_text) > 48 else "")

    with st.chat_message("user"):
        st.markdown(display_text)
        for attachment in attachments:
            _render_attachment(attachment)

    if attachments and any(not attachment.get("uploaded") for attachment in attachments):
        answer = (
            "The attachment upload did not complete, so I cannot review its contents. "
            "Please retry the failed file or use 'Extract handwriting / document text' to preview and download its text."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        chat["messages"].append({"role": "assistant", "content": answer, "meta": None})
        return

    payload = {
        "question": display_text,
        "session_id": chat["session_id"],
        "language": None if language_choice == "Auto" else language_choice.lower(),
        "explanation_mode": st.session_state.get("explanation_mode", "simple"),
    }
    try:
        # This client timeout must stay comfortably above the backend's own
        # LLM-call ceiling (`settings.llm_timeout`, 120s), not equal to it --
        # confirmed live: a real `/chat` call for a cache-miss SECTION_LOOKUP
        # query ("Section 420 IPC") took 126s end-to-end (120s LLM budget +
        # retrieval/fallback overhead) and returned a valid, correctly-
        # grounded answer. At the old `timeout=120`, this client would have
        # already disconnected and shown the generic "couldn't reach the
        # assistant" error below for a request the backend was about to
        # answer correctly -- indistinguishable, from the user's side, from
        # a real failure. 180s leaves real headroom over the backend's own
        # worst case instead of racing it.
        response = auth_client.request(SESSION, API_BASE_URL, "POST", "/chat",
            json=payload, timeout=CLIENT_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 401:
            st.session_state["auth_session_expired"] = True
        response.raise_for_status()
        data = _repair_mojibake(response.json())
    except httpx.HTTPError as exc:
        # Part 50: this used to interpolate the raw exception (`{exc}`) into
        # the user-facing message -- httpx's default str() for a timeout/
        # connection error includes low-level wording ("timed out", host/
        # port details) that has no business reaching an end user. This is
        # a total HTTP-level failure (the backend never even produced a
        # response), distinct from the graceful in-app LLM-provider-error
        # handling already done server-side -- so it needs its own friendly,
        # generic message here rather than assuming that server-side
        # handling covers it.
        friendly_answer = (
            "Sorry, I couldn't reach the assistant just now — this is usually temporary. "
            "Please try sending that again in a moment."
        )
        # Keep the detailed transport failure available to developers without
        # showing host/port details to the user.
        st.session_state["last_chat_error"] = f"{type(exc).__name__}: {exc}"
        with st.chat_message("assistant"):
            st.markdown(friendly_answer)
        chat["messages"].append({"role": "assistant", "content": friendly_answer, "meta": None})
        return

    chat["session_id"] = data.get("session_id") or chat["session_id"]
    answer = data.get("answer", "The assistant did not return an answer.")
    msg_key = f"new{len(chat['messages'])}"
    with st.chat_message("assistant"):
        st.write_stream(_typing_stream(answer))
        _render_assistant_extras(data, msg_key, chat["session_id"], answer)
    chat["messages"].append({"role": "assistant", "content": answer, "meta": data})


# `_try_refresh_admin_token` lived here. It read `admin_token`/
# `admin_refresh_token`, which no UI flow ever set, while the rest of the
# client read `access_token` -- two naming conventions, neither of them
# populated. Both are replaced by `streamlit_app/auth_client.py`, which
# owns `access_token`/`refresh_token` and applies the refresh-and-retry
# rule inside `auth_client.request`.


# CHAT-FIRST REFACTOR: the admin dashboard, the unanswered-question review
# queue, and the Phase-2 "Legal Workflows" workspace used to live here as
# ~300 lines of Streamlit forms behind sidebar buttons. All of it is deleted
# rather than hidden.
#
# Everything those pages did is reachable by talking, routed through
# `POST /chat` -> `app/chatops/orchestrator.py`, which calls the same backend
# services those forms called. The admin REST endpoints are unchanged and
# still enforce their own role checks, so nothing was weakened by removing
# the UI in front of them.

def _handle_voice_turn(chat: dict[str, Any], audio_bytes: bytes, mime_type: str, language_choice: str) -> None:
    """Voice-in/voice-out turn via `/voice/chat`: transcribes the recording,
    runs it through the same chat pipeline as typed input, and gets back
    real spoken audio for the reply (Gemini TTS, see `app/api/voice_router.py` --
    no longer the placeholder stub). Mirrors `_handle_turn`'s message-history
    shape exactly (`role`/`content`/`meta`) so voice and typed turns render
    identically in history; the only addition is `meta["audio_base64"]`,
    which `_render_voice_playback` picks up to auto-play instead of showing
    a "Read aloud" button.
    """
    try:
        response = auth_client.request(SESSION,
            API_BASE_URL,
            "POST",
            "/voice/chat",
            timeout=60,
            files={"audio_file": ("recording.wav", audio_bytes, mime_type)},
            data={
                "session_id": chat["session_id"],
                "language": None if language_choice == "Auto" else language_choice.lower(),
            },
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError:
        st.toast("Couldn't reach the voice assistant — please try again.", icon="⚠️")
        return

    transcribed_text = data.get("transcribed_text", "")
    chat["session_id"] = data.get("session_id") or chat["session_id"]
    chat["messages"].append({"role": "user", "content": transcribed_text, "attachments": []})
    if not chat["title"]:
        chat["title"] = transcribed_text[:48] + ("…" if len(transcribed_text) > 48 else "")
    answer = data.get("ai_response_text", "The assistant did not return an answer.")
    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "audio_base64": data.get("audio_base64"),
            "confidence": data.get("confidence"),
            "detected_intent": data.get("intent"),
            "draft": data.get("draft"),
            "voice_final_confirmation_required": data.get("voice_final_confirmation_required", False),
        },
    })


with st.sidebar:
    st.markdown("### ⚖️ Legal AI Assistant")
    if st.button("＋ New Chat", width="stretch", type="primary"):
        chat = _new_chat()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.active_chat_id = chat["id"]
        st.rerun()

    st.divider()
    for chat_id, chat in reversed(list(st.session_state.chats.items())):
        label = chat["title"] or "New chat"
        is_active = chat_id == st.session_state.active_chat_id
        if st.button(("💬 " if is_active else "") + label, key=f"switch_{chat_id}", width="stretch"):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.divider()
    with st.popover("⚙️ Settings", width="stretch"):
        language_choice = st.selectbox("Language", LANGUAGES, key="language_choice")
        st.toggle("Developer Mode", key="dev_mode", help="Show retrieval, confidence, latency, and source details under each answer.")
        st.selectbox("Explanation", ["simple", "detailed", "advocate"], key="explanation_mode")

    # Authentication must be visible without opening an unrelated settings
    # menu. This remains an identity control, not another feature page: all
    # user/admin capabilities continue to be invoked through the chat.
    account_label = (
        f"👤 {SESSION.get(auth_client.EMAIL_KEY, 'Account')}"
        if auth_client.is_authenticated(SESSION)
        else "🔐 Sign in"
    )
    with st.popover(account_label, width="stretch"):
        if auth_client.is_authenticated(SESSION):
            st.caption("Signed in")
            st.markdown(
                f"**{SESSION.get(auth_client.EMAIL_KEY, 'your account')}** — "
                f"role `{auth_client.current_role(SESSION) or 'user'}`"
            )
            if st.button("Sign out", key="auth_logout", width="stretch"):
                auth_client.logout(SESSION, API_BASE_URL)
                st.session_state.pop("auth_notice", None)
                st.session_state.admin_view = "chat"
                st.rerun()
        else:
            with st.form("auth_login_form", clear_on_submit=True):
                st.caption("Sign in to use your saved documents and any admin capabilities.")
                email = st.text_input("Email", key="auth_email", autocomplete="username")
                # `type="password"` is what keeps the value masked; it is
                # passed straight to `POST /login` and never stored.
                password = st.text_input(
                    "Password", type="password", key="auth_password", autocomplete="current-password"
                )
                if st.form_submit_button("Sign in", width="stretch"):
                    ok, notice = auth_client.login(SESSION, API_BASE_URL, email, password)
                    st.session_state["auth_notice"] = notice
                    if ok:
                        # `clear_on_submit` removes the password widget value;
                        # credentials themselves are never copied into our
                        # authentication state.
                        st.rerun()
            notice = str(SESSION.get("auth_notice") or "")
            if notice:
                st.caption(notice)
        if st.session_state.pop("auth_session_expired", False):
            st.warning("Your session expired. Please sign in again.")

    if auth_client.is_admin(SESSION):
        st.divider()
        if st.session_state.admin_view in {"kb_admin", "coverage_admin", "legal_source_review"}:
            if st.button("💬 Back to chat", key="admin_view_chat", width="stretch"):
                st.session_state.admin_view = "chat"
                st.rerun()
        else:
            if st.button("🗂️ KB Review (Admin)", key="admin_view_kb", width="stretch"):
                st.session_state.admin_view = "kb_admin"
                st.rerun()
            if st.button("🗺️ State/UT Coverage (Admin)", key="admin_view_coverage", width="stretch"):
                st.session_state.admin_view = "coverage_admin"
                st.rerun()
            if st.button("⚖️ Legal Source Review (Admin)", key="admin_view_legal_review", width="stretch"):
                st.session_state.admin_view = "legal_source_review"
                st.rerun()

    st.text_input("Search this chat", key="chat_search", placeholder="Search messages")

# CHAT-FIRST REFACTOR
# -------------------
# The sidebar used to carry a button per feature -- Legal Workflows, Verify a
# Document, Notary Queue, Notarization Admin, Saved drafts, Download center,
# Admin Dashboard -- each opening its own page with its own forms. All of it
# is gone, and none of it was merely hidden: every one of those capabilities
# is now reachable by TALKING to the assistant, routed by
# `app/chatops/orchestrator.py` from `POST /chat`.
#
#   "meri saved drafts dikhao"        -> saved_drafts workflow
#   "PDF me download karna hai"       -> draft_export workflow, inline link
#   "mujhe cyber fraud report karna hai" -> cyber_fraud workflow
#   "jurisdiction check karo"         -> jurisdiction workflow
#   "document verify karo"            -> notarization_verify workflow
#   "meri pending review requests"    -> notary_queue (role-gated)
#   "notarization audit log dikhao"   -> notary_admin (role-gated)
#
# What the sidebar keeps is exactly what the spec allows: new chat, previous
# conversations, and minimal settings. The composer keeps text, send, file
# attachment and voice.
#
# The specialised REST endpoints all still exist and still work -- this is a
# frontend change, not an API removal, so existing integrations are
# unaffected.

if st.session_state.admin_view == "kb_admin" and auth_client.is_admin(SESSION):
    kb_admin_page.render(SESSION, API_BASE_URL)
    st.stop()
elif st.session_state.admin_view == "coverage_admin" and auth_client.is_admin(SESSION):
    coverage_admin_page.render(SESSION, API_BASE_URL)
    st.stop()
elif st.session_state.admin_view == "legal_source_review" and auth_client.is_admin(SESSION):
    legal_source_review_page.render(SESSION, API_BASE_URL)
    st.stop()
elif st.session_state.admin_view in {"kb_admin", "coverage_admin", "legal_source_review"}:
    # Role was lost mid-session (sign-out, expired token) -- fall back to
    # chat instead of rendering an admin page for a non-admin session.
    st.session_state.admin_view = "chat"

language_choice = st.session_state.get("language_choice", "Auto")
active_chat = _active_chat()

document_extraction_ui.render(SESSION, API_BASE_URL, active_chat["session_id"])

if not active_chat["messages"]:
    st.markdown("<h2 style='text-align:center; margin-top:3rem;'>How can I help with your legal question?</h2>", unsafe_allow_html=True)
    _render_draft_entry_actions(active_chat["session_id"])
    st.divider()
    cols = st.columns(2)
    for index, example in enumerate(EXAMPLE_PROMPTS):
        with cols[index % 2]:
            if st.button(example, key=f"example_{index}", width="stretch"):
                _send_chip(example)
else:
    _render_history(active_chat)

# Streamlit's `st.chat_input` renders its send arrow via its own built-in
# component -- there's no way to inject a custom button literally inside it.
# The closest available equivalent to "voice controls next to the send
# button" is a slim, right-aligned toolbar sitting directly above the input
# box (mirroring where the send arrow sits, on the right edge), instead of
# the previous wide "🎤 Voice" popover that read as a separate, disconnected
# feature elsewhere on the page.
_, mic_col, speak_col = st.columns([10, 1, 1])
with mic_col, st.popover("🎤", help="Ask by voice instead of typing"):
    st.caption("Record your question — it's transcribed and answered like a typed message, with a spoken reply.")
    voice_recording = st.audio_input("Speak your question", key="voice_recorder", label_visibility="collapsed")
with speak_col:
    _last_assistant_message = next(
        (message for message in reversed(active_chat["messages"]) if message["role"] == "assistant"), None
    )
    if st.button("🔊", key="explain_last_reply", help="Have the last reply read aloud", disabled=_last_assistant_message is None):
        _meta = _last_assistant_message.get("meta") or {}
        if not _meta.get("audio_base64"):
            try:
                speak_response = auth_client.request(SESSION,
                    API_BASE_URL,
                    "POST",
                    "/voice/speak",
                    timeout=30,
                    json={"text": _last_assistant_message["content"]},
                )
                speak_response.raise_for_status()
                _meta["audio_base64"] = speak_response.json().get("audio_base64")
                _last_assistant_message["meta"] = _meta
            except httpx.HTTPError:
                st.toast("Couldn't generate audio right now — please try again.", icon="⚠️")
        st.rerun()

prompt = st.chat_input(
    "Message Legal AI Assistant…",
    accept_file="multiple",
    file_type=UPLOAD_FILE_TYPES,
)
pending = st.session_state.pop("pending_input", None)

new_text, new_files = "", []
if prompt is not None:
    new_text, new_files = prompt.text, prompt.files
elif pending is not None:
    new_text = pending

if new_text.strip() or new_files:
    _handle_turn(active_chat, new_text, new_files, language_choice)
elif voice_recording is not None:
    # `st.audio_input` keeps returning the SAME recording across reruns
    # until the user records a new one -- without this guard, every rerun
    # this session triggers (a chip click, a sidebar toggle, anything)
    # would silently resend the last recording and duplicate the turn.
    # Hashed content (not just object identity) survives Streamlit's
    # session-state serialization the same way `pending_input` doesn't
    # need to, since audio bytes are cheap to hash and don't change
    # spuriously between reruns of the same recording.
    audio_bytes = voice_recording.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.get("last_voice_hash") != audio_hash:
        st.session_state.last_voice_hash = audio_hash
        _handle_voice_turn(active_chat, audio_bytes, voice_recording.type or "audio/wav", language_choice)
        st.rerun()
