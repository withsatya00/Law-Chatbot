"""Secure artifact references for files produced inside a conversation.

The rule this module exists to enforce: a chat reply never carries a
filesystem path. It carries an opaque reference the API can resolve back to a
file, for the owner, through the existing download endpoints.

Why that matters here specifically: workflows call exporters that return
`Path` objects on the server. Putting one of those into an assistant message
would leak the server's directory layout into text that gets logged, cached,
rendered, and sometimes copied into a support ticket -- and would invite a
client to request an arbitrary path.
"""

from dataclasses import asdict, dataclass
from typing import Any

# Media types for the formats the drafting/notarization exporters produce.
MEDIA_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "rtf": "application/rtf",
}


@dataclass(frozen=True)
class ChatArtifact:
    """A downloadable file, described without exposing where it lives.

    `download_path` is an API ROUTE (e.g. `/draft/export`), not a file path,
    and the route re-checks ownership when it is called -- possessing this
    reference is not itself authorisation.
    """

    kind: str            # "draft" | "notarized_document" | "report"
    label: str           # what to show on the link
    fmt: str             # pdf/docx/txt/rtf
    download_path: str   # API route the client calls
    document_id: str = ""
    draft_id: str = ""
    media_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media_type"] = self.media_type or MEDIA_TYPES.get(self.fmt, "application/octet-stream")
        return data


def draft_artifact(draft_id: str, fmt: str, label: str = "") -> dict[str, Any]:
    """A reference to an exported draft.

    Points at the existing `/draft/export` route, which already enforces
    owner/session scoping -- this adds no new download surface.
    """
    return ChatArtifact(
        kind="draft",
        label=label or f"Download {fmt.upper()}",
        fmt=fmt,
        download_path="/draft/export",
        draft_id=draft_id,
    ).as_dict()


def notarized_artifact(document_id: str, fmt: str = "pdf", label: str = "") -> dict[str, Any]:
    """A reference to a NOTARIZED document.

    The route it names refuses any document that is not actually notarized,
    so this reference cannot be used to pull a draft dressed up as a
    notarized file.
    """
    return ChatArtifact(
        kind="notarized_document",
        label=label or "Download notarized PDF",
        fmt=fmt,
        download_path=f"/notarization/documents/{document_id}/download",
        document_id=document_id,
    ).as_dict()
