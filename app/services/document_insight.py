"""Reading a user's own uploaded document, with page evidence.

The shared foundation under the document workflows: fetch the indexed text of
one document, ownership-checked, keeping each chunk's page number so a
finding can say WHERE it was read. Everything that reports on a document --
the summary, the risky-clause list, the review checklist, the two-document
comparison -- resolves its document through `load_document` here, which means
a single place enforces "may this person see this file".

Page numbers are never invented. A chunk indexed before page capture existed,
or a format with no page identity at all (plain text, HTML), yields `None`,
and the caller must present that as "page not recorded" rather than page 1.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.exceptions import BadRequestError, NotFoundError
from app.llm.base import ChatMessage
from app.llm.deadline import call_with_hard_timeout
from app.llm.factory import LLMFactory
from app.llm.prompts import prompt_registry
from app.repositories.documents import EmbeddingMetadataRepository
from app.schemas.document_transcribe import DocumentTranscribeResponse
from app.services.document_service import ensure_document_access
from app.utils.prompt_security import PromptInjectionScanner


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    page_number: int | None = None


@dataclass
class LoadedDocument:
    """One document's indexed text, in order, with page provenance."""

    document_id: str
    filename: str
    language: str = ""
    chunks: list[DocumentChunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(chunk.text for chunk in self.chunks)

    @property
    def has_page_evidence(self) -> bool:
        return any(chunk.page_number is not None for chunk in self.chunks)

    def page_index(self) -> dict[str, int]:
        """`{sentence prefix: page number}` for the extractors that want pages.

        Only chunks that actually carry a page number appear, so a document
        indexed without page capture produces an empty index and every
        downstream `source_page` stays `None` rather than defaulting to 1.
        """
        index: dict[str, int] = {}
        for chunk in self.chunks:
            if chunk.page_number is None:
                continue
            for sentence in re.split(r"(?<=[.\n।])\s+", chunk.text):
                key = sentence.strip()[:40]
                if len(key) >= 12:
                    index.setdefault(key, chunk.page_number)
        return index

    def page_for(self, snippet: str) -> int | None:
        """The page a snippet was read from, or None if it cannot be pinned.

        Matching is on a normalised prefix of the snippet, because the text a
        summariser quotes is rarely byte-identical to the indexed text. A
        snippet that matches nothing, or whose chunk carries no page number,
        returns None -- which the caller must render as "page not recorded".
        """
        needle = _normalise(snippet)[:80]
        if not needle:
            return None
        for chunk in self.chunks:
            if needle in _normalise(chunk.text):
                return chunk.page_number
        return None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


async def load_document(
    document_id: str, authenticated_user_id: str | None, session_id: str | None
) -> LoadedDocument:
    """The indexed text of one document, after an ownership check.

    Raises `NotFoundError` when nothing is indexed under this id and
    `ForbiddenError` when it belongs to someone else -- deliberately
    distinguished, matching `CaseService._get_owned_case`, because "no such
    document" and "not yours" are different facts and conflating them makes
    a real bug (an unindexed upload) indistinguishable from a refusal.
    """
    repository = EmbeddingMetadataRepository()
    cursor = repository.collection.find(
        {"document_id": document_id}, {"text": 1, "metadata": 1, "chunk_index": 1}
    )
    rows: list[dict[str, Any]] = [row async for row in cursor]
    if not rows:
        raise NotFoundError("That document is not available.")

    # Ownership fields are document-wide, so the first chunk's metadata is
    # representative -- the same assumption `DocumentService.analyze` makes.
    first_metadata: dict[str, Any] = rows[0].get("metadata") or {}
    ensure_document_access(first_metadata, authenticated_user_id, session_id)

    rows.sort(key=lambda row: int(row.get("chunk_index") or 0))
    chunks = [
        DocumentChunk(
            text=str(row.get("text", "")),
            page_number=_page_number((row.get("metadata") or {}).get("page_number")),
        )
        for row in rows
        if str(row.get("text", "")).strip()
    ]
    if not chunks:
        raise BadRequestError("That document has no readable text.")
    return LoadedDocument(
        document_id=document_id,
        filename=str(first_metadata.get("source_document") or first_metadata.get("filename") or "document"),
        language=str(first_metadata.get("language", "")),
        chunks=chunks,
    )


def _page_number(value: Any) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


@dataclass(frozen=True)
class AnsweredQuestion:
    """One question, answered strictly from a single `LoadedDocument`.

    `found=False` means the model looked and the document genuinely does
    not answer the question -- distinct from `answer_question` returning
    `None`, which means the model call itself failed and nothing was
    determined either way. Callers must never collapse those two into the
    same "couldn't find it" message: one is a fact about the document, the
    other is a transient failure the user should be told to retry.
    """

    found: bool
    answer: str = ""
    quote: str = ""
    page: int | None = None


def _render_with_pages(document: LoadedDocument, max_chars: int) -> str:
    """The document's own text, each chunk prefixed with the page it was
    read from, bounded to `max_chars` -- the same length discipline
    `DocumentService._llm_analysis` already applies (`settings.
    document_analysis_max_chars`), for the same reason: an unbounded upload
    must not blow the prompt budget or the request's own wall-clock deadline.
    """
    parts: list[str] = []
    total = 0
    for chunk in document.chunks:
        marker = f"[Page {chunk.page_number}]" if chunk.page_number is not None else "[Page not recorded]"
        piece = f"{marker}\n{chunk.text}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece) + 2
    return "\n\n".join(parts)


async def answer_question(document: LoadedDocument, question: str, language: str) -> AnsweredQuestion | None:
    """Answers ONE specific question strictly from `document`'s own indexed
    text -- never the shared knowledge base, never a different document the
    same conversation may have uploaded, and never general legal knowledge
    standing in for content that is not actually there.

    The page cited is never taken on the model's own say-so: `document.
    page_for(quote)` independently locates the returned quote in the
    document's own chunk text, the same deterministic check
    `RiskyClauseWorkflow` already applies to a clause it reports -- a model
    that names the wrong page number for a real quote still gets the RIGHT
    page here, and a hallucinated quote that matches nothing gets no page
    number attached rather than a wrong one.

    Returns `None` on an LLM failure/timeout -- bounded by the same
    `call_with_hard_timeout` backstop every other drafting/chat LLM call
    now goes through (QA 2026-09-11/12) -- so the caller can tell "the model
    looked and it genuinely is not there" (`found=False`) apart from "the
    call itself did not work" (`None`), and never reports the latter as the
    former.
    """
    document_text = _render_with_pages(document, settings.document_analysis_max_chars)
    prompt = prompt_registry.render(
        "document_question_prompt",
        question=question,
        language=language or "the document's language",
        document_text=document_text,
    )
    llm = LLMFactory.create_resilient()
    try:
        response = await call_with_hard_timeout(
            llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0),
            fallback_seconds=settings.draft_generation_budget_seconds,
        )
    except TimeoutError:
        return None
    if response.error or not response.content.strip():
        return None
    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    found = bool(parsed.get("found_in_document"))
    answer = str(parsed.get("answer") or "").strip()
    quote = str(parsed.get("supporting_quote") or "").strip()
    if not found or not answer:
        return AnsweredQuestion(found=False)
    return AnsweredQuestion(found=True, answer=answer, quote=quote, page=document.page_for(quote) if quote else None)


@dataclass(frozen=True)
class TranscribeResult:
    """Wraps `DocumentTranscribeResponse` with the one extra fact the
    workflow layer needs that the model itself must never decide:
    whether the source text tripped the prompt-injection scanner.

    Unlike `DocumentService.analyze` (which refuses the whole request when
    its scanner fires -- appropriate for a pasted, one-off excerpt), a
    scanned legal document is exactly the input this feature exists to
    handle, and an applicant's own document can legitimately quote
    threatening or unusual language. The prompt itself already instructs
    the model to never execute or obey text found in the document, and the
    response is constrained to fixed JSON fields, so refusing outright
    would block the feature's core use case for a risk that structural
    output-shape already contains. Instead, a positive scan surfaces as a
    warning the caller can show, not a hard failure.
    """

    response: DocumentTranscribeResponse
    injection_suspected: bool = False


async def transcribe_document(document: LoadedDocument, language: str) -> TranscribeResult | None:
    """Retypes `document` into a clean, professionally formatted version,
    preserving its own facts, language and script -- see
    `document_transcribe_prompt.md` for the full no-fabrication contract.

    Returns `None` on an LLM failure/timeout/invalid-JSON, the same
    "could not determine, do not fabricate" signal `answer_question` uses
    for the same reason: a caller must never render a guessed transcription
    as if it were a real one.
    """
    document_text = _render_with_pages(document, settings.document_analysis_max_chars)
    injection_suspected, _findings = PromptInjectionScanner().scan(document_text)
    prompt = prompt_registry.render(
        "document_transcribe_prompt",
        language=language or "the document's language",
        document_text=document_text,
    )
    llm = LLMFactory.create_resilient()
    try:
        response = await call_with_hard_timeout(
            llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0),
            fallback_seconds=settings.draft_generation_budget_seconds,
        )
    except TimeoutError:
        return None
    if response.error or not response.content.strip():
        return None
    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        validated = DocumentTranscribeResponse.model_validate(parsed)
    except ValueError:
        return None
    if not validated.formatted_text.strip():
        return None
    return TranscribeResult(response=validated, injection_suspected=injection_suspected)
