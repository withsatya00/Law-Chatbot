"""Pure formatting for the band of secondary information under a chat answer.

Kept out of `streamlit_app/app.py` so it can be tested directly: importing
`app.py` runs `st.set_page_config`, injects CSS and touches `st.session_state`,
none of which works outside a live Streamlit script run. What a reader is told
about the sources behind a legal answer is worth a test, so it lives here and
`app.py` imports it.
"""

from typing import Any


def _looks_like_refusal(answer: str) -> bool:
    """Whether `answer` opens with the strict-RAG refusal, in any language.

    The backend owns that wording (`app.core.constants`), so it is read from
    there rather than restated here. The import is deferred and optional
    because this module also runs inside the Streamlit process, where
    `streamlit_app/` is on `sys.path` and `app` resolves to `app.py`, not to
    the backend package -- there, only the explicit `no_verified_context`
    flag is available, which is the signal the API sets on every response it
    builds today.
    """
    try:
        from app.core.constants import is_no_verified_context
    except Exception:  # noqa: BLE001 - the flag above is the primary signal
        return False
    return bool(is_no_verified_context(answer))


def suppresses_apparatus(data: dict[str, Any]) -> bool:
    """True when this reply is the strict-RAG refusal.

    A refusal says no verified document supports an answer, so rendering a
    Sources list, an applicable-law line or a "how current are these
    sources?" note beside it contradicts the reply itself -- which is what a
    reported turn did, showing BNS and GST citations under "no verified
    document related to this question is available in the Knowledge Base."

    The API is authoritative: `app/services/safe_decline.py` empties those
    fields before the response is built or stored. This is defence in depth
    for a payload that predates that invariant -- a message replayed from
    chat history, or a cached entry written by an older build.
    """
    if data.get("no_verified_context"):
        return True
    return _looks_like_refusal(str(data.get("answer") or ""))


def source_lines(data: dict[str, Any]) -> list[str]:
    """The citation lines for one answer: the provision, the document, the page
    evidence where the source actually has it, and its verification status.

    Post-Phase-3 hardening (Phase 2, milestone E). Sources used to be rendered
    only inside the developer panel, which is off by default -- so the Act and
    section behind an answer, which the backend computes and returns on every
    grounded reply, were invisible to every ordinary user, while the capability
    overview told them a citation would be there.

    Nothing here derives a citation from the answer text: `applicable_law`,
    `label` and `evidence_pages` are all built server-side from source
    metadata (see `app/rag/citation.py`). A source with no Act or section
    contributes its document name and nothing more, which is the honest
    rendering of "this is where it came from, and the provision was not
    recorded" -- never a guess at one. A page appears only when the server sent
    one; `evidence_pages` omits sources with no page identity rather than
    reporting page 1.
    """
    if suppresses_apparatus(data):
        return []

    lines: list[str] = []
    for provision in data.get("applicable_law") or []:
        lines.append(f"**{provision}**")

    pages_by_document: dict[str, list[str]] = {}
    for page in data.get("evidence_pages") or []:
        document = str(page.get("source_document") or "source")
        label = page.get("label") or (
            f"p. {page['page_number']}" if page.get("page_number") is not None else ""
        )
        if label:
            pages_by_document.setdefault(document, []).append(str(label))

    for source in data.get("sources") or []:
        document = str(source.get("source_document") or "")
        label = source.get("label") or document or "Unattributed source"
        parts = [str(label)]
        pages = ", ".join(pages_by_document.get(document, []))
        if pages:
            parts.append(pages)
        # Said plainly rather than with a reassuring badge: an unverified
        # source is the common case in this corpus, and hiding that would make
        # the rare verified one indistinguishable from it.
        status = source.get("verification_status")
        if status and status != "verified":
            parts.append(str(status).replace("_", " "))
        entry = " · ".join(parts)
        url = source.get("url")
        lines.append(f"- [{entry}]({url})" if url else f"- {entry}")
    return lines
