"""User-facing wording for a tracked shared-KB upload."""

from typing import Any

TERMINAL_UPLOAD_STATUSES = frozenset({"indexed", "duplicate", "failed", "needs_review"})


def status_note(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "pending").lower()
    staging_id = str(payload.get("staging_id") or "")
    generated = str(payload.get("generated_filename") or payload.get("original_filename") or "document")
    reason = str(payload.get("reason") or "").strip()
    chunks = payload.get("chunks_indexed")

    if status == "indexed":
        detail = f" ({chunks} chunks)" if isinstance(chunks, int) else ""
        # Phase 1 "Jurisdiction-Aware Knowledge Base": a document can finish
        # indexing (staging status "indexed") while still being excluded from
        # shared search because its jurisdiction metadata is missing or
        # unverified (`review_status`, set by `kb_jurisdiction.normalize_jurisdiction`).
        # Silence here would tell an admin the document is live when it isn't.
        if str(payload.get("review_status") or "") == "needs_review":
            reasons = payload.get("review_reasons") or []
            reason_text = f" ({'; '.join(str(r) for r in reasons)})" if reasons else ""
            return (
                f"Indexed as {generated}{detail}, but NOT yet visible in shared search: "
                f"jurisdiction metadata needs review{reason_text}."
            )
        return f"Indexed into the shared Knowledge Base as {generated}{detail}."
    if status == "duplicate":
        detail = f" {reason}" if reason else ""
        return f"Not indexed again because this is a duplicate.{detail}".strip()
    if status == "failed":
        detail = f" Reason: {reason}" if reason else ""
        return f"Knowledge Base indexing failed.{detail} Ask for failed KB files to review it."
    if status == "needs_review":
        detail = f" Reason: {reason}" if reason else ""
        return f"Waiting for administrator review.{detail}"
    if status == "processing":
        return "Shared Knowledge Base indexing is in progress."
    tracking = f" Tracking ID: {staging_id}." if staging_id else ""
    return f"Queued for shared Knowledge Base indexing.{tracking} Status will refresh automatically."
