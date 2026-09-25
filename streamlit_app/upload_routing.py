"""Choose the attachment destination from an explicit chat instruction.

Ordinary attachments are private conversation documents. Sending bytes to
the shared legal Knowledge Base is a materially different admin action, so it
requires both an explicit KB-upload phrase and an admin/super-admin session;
the API still performs the authoritative JWT role check.
"""

import re
from typing import Literal

UploadTarget = Literal["private", "admin_kb", "admin_required"]


def upload_error_note(response) -> str:
    """Explain known transient failures without exposing arbitrary provider bodies."""
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict) and (
            error.get("code") == "handwriting_unavailable"
            or str(error.get("message", "")).startswith("Handwriting OCR provider returned HTTP 503")
        ):
            return "Not uploaded: the handwriting service is temporarily unavailable. Please retry shortly."
    except (ValueError, AttributeError):
        pass
    if response.status_code in {429, 502, 503, 504}:
        return "Not uploaded: the service is temporarily busy or unavailable. Please retry shortly."
    return f"Could not upload this file (server returned {response.status_code})."

_KB = re.compile(r"\b(?:kb|knowledge\s*base|knowledgebase)\b", re.IGNORECASE)
_UPLOAD = re.compile(
    r"\b(?:upload|add|ingest|index|import|daal(?:o|na)?|dalo|bhejo|jodo|शामिल|अपलोड)\w*\b",
    re.IGNORECASE,
)


def upload_target(message: str, role: str) -> UploadTarget:
    explicit_kb_upload = bool(_KB.search(message or "") and _UPLOAD.search(message or ""))
    if not explicit_kb_upload:
        return "private"
    if role in {"admin", "super_admin"}:
        return "admin_kb"
    return "admin_required"
